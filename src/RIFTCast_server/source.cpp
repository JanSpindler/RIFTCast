#include <iostream>

#include <ATCG.h>
#include <Core/EntryPoint.h>

#include <thread>
#include <mutex>
#include <queue>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>

#include <riftcast/DatasetImporter.h>
#include <maskcompression/decompress.h>
#include <torchhull/visual_hull.h>
#include <riftcast/riftcastkernels.h>
#include <zstd.h>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#include <torch/torch.h>
#include <glad/glad.h>

#include <riftcast/protocol/riftcastprotocol.h>

#include <riftcast/GeometryModule.h>
#include <riftcast/RenderModule.h>
#include <riftcast/InpaintingModule.h>

//#include <interceptor/logging.h>

#include <zmq.hpp>
#include <json.hpp>

using json = nlohmann::json;

struct SMPLXResult
{
    // Metadata
    int frame_index;
    int num_vertices;
    int num_faces;
    
    // SMPL-X parameters (182 floats total)
    std::vector<float> global_orient;      // 3 floats
    std::vector<float> body_pose;          // 63 floats
    std::vector<float> left_hand_pose;     // 45 floats
    std::vector<float> right_hand_pose;    // 45 floats
    std::vector<float> jaw_pose;           // 3 floats
    std::vector<float> betas;              // 10 floats
    std::vector<float> expression;         // 10 floats
    std::vector<float> transl;             // 3 floats
    
    // Mesh data
    std::vector<float> vertices;           // num_vertices * 3
    std::vector<int32_t> faces;            // num_faces * 3
};

class SMPLXClient
{
public:
    SMPLXClient(const std::string& address = "tcp://localhost:5555")
        : context(1), socket(context, zmq::socket_type::req)
    {
        int hwm = 10;
        socket.setsockopt(ZMQ_SNDHWM, &hwm, sizeof(int));
        socket.setsockopt(ZMQ_RCVHWM, &hwm, sizeof(int));
    
        socket.connect(address);
        std::cout << "Connected to SMPL-X server at " << address << std::endl;
    }

    bool initialize(const std::string& vci_dir)
    {
        json metadata = {
            {"command", "initialize"},
            {"vci_dir", vci_dir}
        };

        std::string metadata_str = metadata.dump();
        socket.send(zmq::buffer(metadata_str), zmq::send_flags::none);

        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);

        auto response = json::parse(std::string(static_cast<char*>(reply.data()), reply.size()));

        if (response["status"] == "success")
        {
            std::cout << "SMPL-X session initialized successfully." << std::endl;
            return true;
        }
        else
        {
            std::cerr << "Failed to initialize SMPL-X session: " << response["error"] << std::endl;
            return false;
        }
    }

    void process_frame(
        const std::map<std::string, torch::Tensor>& image_tensors,
        const std::vector<std::string>& selected_cam_ids,
        SMPLXResult& result)
    {
        // Build metadata
        std::vector<std::string> cam_ids;
        for (const auto& [cam_id, _] : image_tensors)
        {
            cam_ids.push_back(cam_id);
        }
        
        json metadata = {
            {"command", "process_frame"},
            {"cam_ids", cam_ids},
            {"num_images", cam_ids.size()}
        };
        
        if (!selected_cam_ids.empty()) 
        {
            metadata["selected_cam_ids"] = selected_cam_ids;
        }
        
        // Send metadata (part 0)
        std::string metadata_str = metadata.dump();
        socket.send(zmq::buffer(metadata_str), zmq::send_flags::sndmore);
        
        // Send image data (parts 1+)
        size_t idx = 0;
        for (const std::string& cam_id : cam_ids) 
        {
            // Parse tensor
            const torch::Tensor& tensor = image_tensors.at(cam_id);
            const torch::Tensor tensor_cpu = tensor.to(torch::kCPU).to(torch::kUInt8).contiguous();

            const auto shape = tensor_cpu.sizes();
            int height = shape[0];
            int width  = shape[1];
            int channels = shape[2];
            
            int shape_data[3] = {height, width, channels};
            socket.send(zmq::buffer(shape_data, sizeof(shape_data)), zmq::send_flags::sndmore);
            
            // Send image bytes
            const zmq::send_flags flags = (idx < cam_ids.size() - 1) ? zmq::send_flags::sndmore : zmq::send_flags::none;
            socket.send(zmq::buffer(tensor_cpu.data_ptr(), tensor_cpu.numel()), flags);
            ++idx;
        }
        
        // Receive multipart response
        std::vector<zmq::message_t> responses;
        int more;
        size_t more_size = sizeof(more);
        
        do {
            zmq::message_t msg;
            socket.recv(msg, zmq::recv_flags::none);
            socket.getsockopt(ZMQ_RCVMORE, &more, &more_size);
            responses.push_back(std::move(msg));
        } while (more);
        
        if (responses.size() == 1) {
            // Error response (JSON)
            std::string error_str(static_cast<char*>(responses[0].data()), responses[0].size());
            auto error_json = json::parse(error_str);
            
            // Return empty result on error (or could throw)
            std::cerr << "Server error: " << error_json["message"].get<std::string>() << std::endl;
            result.frame_index = -1;
            result.num_vertices = 0;
            result.num_faces = 0;
        }
        else if (responses.size() == 4) {
            // Binary response
            // Part 0: Metadata (JSON)
            std::string metadata_str(static_cast<char*>(responses[0].data()), responses[0].size());
            auto metadata = json::parse(metadata_str);
            
            result.frame_index = metadata["frame_index"];
            result.num_vertices = metadata["num_vertices"];
            result.num_faces = metadata["num_faces"];
            
            // Part 1: SMPL-X params (182 float32)
            const float* params_ptr = static_cast<const float*>(responses[1].data());
            size_t params_count = responses[1].size() / sizeof(float);
            
            // Parse params (offsets based on concatenation order)
            size_t offset = 0;
            result.global_orient.assign(params_ptr + offset, params_ptr + offset + 3);
            offset += 3;
            
            result.body_pose.assign(params_ptr + offset, params_ptr + offset + 63);
            offset += 63;
            
            result.left_hand_pose.assign(params_ptr + offset, params_ptr + offset + 45);
            offset += 45;
            
            result.right_hand_pose.assign(params_ptr + offset, params_ptr + offset + 45);
            offset += 45;
            
            result.jaw_pose.assign(params_ptr + offset, params_ptr + offset + 3);
            offset += 3;
            
            result.betas.assign(params_ptr + offset, params_ptr + offset + 10);
            offset += 10;
            
            result.expression.assign(params_ptr + offset, params_ptr + offset + 10);
            offset += 10;
            
            result.transl.assign(params_ptr + offset, params_ptr + offset + 3);
            
            // Part 2: Mesh vertices (num_vertices * 3 float32)
            const float* vertices_ptr = static_cast<const float*>(responses[2].data());
            size_t vertices_count = responses[2].size() / sizeof(float);
            result.vertices.assign(vertices_ptr, vertices_ptr + vertices_count);
            
            // Part 3: Mesh faces (num_faces * 3 int32)
            const int32_t* faces_ptr = static_cast<const int32_t*>(responses[3].data());
            size_t faces_count = responses[3].size() / sizeof(int32_t);
            result.faces.assign(faces_ptr, faces_ptr + faces_count);
            
            // std::cout << "Frame " << result.frame_index << " processed successfully" << std::endl;
            // std::cout << "Vertices: " << result.num_vertices << ", Faces: " << result.num_faces << std::endl;            
        }
        else {
            std::cerr << "Unexpected response format: " << responses.size() << " parts" << std::endl;
            result.frame_index = -1;
            result.num_vertices = 0;
            result.num_faces = 0;
        }
    }

    void reset() 
    {
        json metadata = {{"command", "reset"}};
        std::string metadata_str = metadata.dump();
        socket.send(zmq::buffer(metadata_str), zmq::send_flags::none);
        
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);
        auto response = json::parse(std::string(static_cast<char*>(reply.data()), reply.size()));
        std::cout << "Reset: " << response["message"] << std::endl;
    }

private:
    zmq::context_t context;
    zmq::socket_t socket;
};

struct ClientState
{
    ClientState() = default;

    ClientState(std::function<void(ClientState*)> render_function,
                std::function<void(ClientState*)> inpaint_function,
                const atcg::ref_ptr<rift::DatasetImporter>& dataloader)
    {
        running        = true;
        render_thread  = std::thread(render_function, this);
        inpaint_thread = std::thread(inpaint_function, this);
    }

    void stop()
    {
        running = false;
        if(render_thread.joinable()) render_thread.join();
        if(inpaint_thread.joinable()) inpaint_thread.join();
    }

    ~ClientState() { /*render_module.reset();*/ }

    // Data that is exclusive to the render thread
    // atcg::ref_ptr<rift::RenderModule> render_module;

    // Input to the rendering thread
    glm::mat4 render_input_view;
    glm::mat4 render_input_projection;
    uint32_t render_input_width;
    uint32_t render_input_height;

    // Output of rendering thread (Input for inpainting thread)
    torch::Tensor render_output_img;
    torch::Tensor render_output_mask;
    torch::Tensor render_output_depth;
    uint32_t render_output_frame_idx;
    glm::mat4 render_output_inv_view_projection;

    // Memory exlcusive for inpainting thread

    // Memory holding the responses. Updated by the inpainting thread
    std::vector<uint8_t> last_response;    // The last sent response (finished frame)
    std::vector<uint8_t> current_response;

    bool inpaint_done   = false;
    bool rendering_done = true;

    // Threads
    std::atomic_bool running = false;

    std::thread render_thread;
    std::mutex render_mutex;
    std::atomic_bool start_rendering = false;

    std::thread inpaint_thread;
    std::mutex inpaint_mutex;
    std::atomic_bool start_inpaint = false;
};

class RIFTCastServer : public atcg::Application
{
public:
    RIFTCastServer(const atcg::WindowProps& props) : atcg::Application(props) {}

    ~RIFTCastServer() {}

    struct DataPacket
    {
        DataPacket() = default;

        DataPacket(const std::vector<uint8_t>& data, const uint64_t client_id) : data(data), client_id(client_id) {}

        std::vector<uint8_t> data;
        uint64_t client_id;
    };

    void process_connect(const uint64_t client_id)
    {
        client_states.insert(
            std::make_pair(client_id,
                           atcg::make_ref<ClientState>([this](ClientState* state) { this->render(state); },
                                                       [this](ClientState* state) { this->inpaint(state); },
                                                       dataloader)));
    }

    void process_disconnect(const uint64_t client_id)
    {
        auto client = client_states[client_id];
        client->stop();
        client_states.erase(client_id);
    }

    std::vector<uint8_t> process_render(uint8_t* data, uint32_t& offset, const uint64_t client_id)
    {
        atcg::Timer timer;
        // 1. Get client state
        std::shared_ptr<ClientState> state = client_states[client_id];

        if(!state)
        {
            return rift::protocol::createNoUpdateMessage();
        }

        {
            std::lock_guard guard(state->render_mutex);
            if(state->rendering_done)
            {
                state->rendering_done = false;

                // 2. Get camera parameters
                state->render_input_width  = atcg::NetworkUtils::readInt<uint32_t>(data, offset);
                state->render_input_height = atcg::NetworkUtils::readInt<uint32_t>(data, offset);

                offset += sizeof(uint32_t);
                float* view_data = (float*)(data + offset);
                offset += sizeof(glm::mat4);

                offset += sizeof(uint32_t);
                float* projection_data = (float*)(data + offset);
                offset += sizeof(glm::mat4);

                state->render_input_view       = glm::make_mat4(view_data);
                state->render_input_projection = glm::make_mat4(projection_data);

                state->start_rendering = true;
            }
        }

        {
            std::lock_guard guard(state->inpaint_mutex);
            if(state->inpaint_done)
            {
                // At this point, the inpainting thread is idling or already working on the next frame, therefore ->
                // mutex on output
                state->inpaint_done = false;
                std::swap(state->last_response, state->current_response);
            }
            else
            {
                state->last_response = rift::protocol::createNoUpdateMessage();
            }
        }


        return state->last_response;
    }

    std::vector<uint8_t> process_masks()
    {
        auto cam_valid = torch::ones({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());
        auto masks     = dataloader->getMasks(cam_valid);

        size_t message_size = sizeof(rift::protocol::ProtocolHeader) + 3 * sizeof(uint32_t);

        for(auto& mask: masks)
        {
            message_size += mask.element_size() * mask.numel() + sizeof(uint32_t);
        }

        std::vector<uint8_t> response(message_size);
        auto header = rift::protocol::ProtocolHeader(rift::protocol::MessageTask::MASKS);
        std::memcpy(response.data(), &header, sizeof(rift::protocol::ProtocolHeader));
        uint32_t offset = sizeof(rift::protocol::ProtocolHeader);

        atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, (uint32_t)masks.size());
        atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, dataloader->width());
        atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, dataloader->height());


        for(auto& mask: masks)
        {
            atcg::NetworkUtils::writeBuffer(response.data(),
                                            offset,
                                            (uint8_t*)mask.cpu().data_ptr(),
                                            mask.numel() * mask.element_size());
        }

        return response;
    }

    std::vector<uint8_t> process_cameras()
    {
        std::vector<glm::mat4> views;
        std::vector<glm::mat4> projections;

        const auto& cameras = dataloader->getCameras();

        for(auto& camera: cameras)
        {
            views.push_back(camera.cam->getView());
            projections.push_back(camera.cam->getProjection());
        }

        return rift::protocol::createCameraResponse(views, projections);
    }

    std::vector<uint8_t> process_masks_images()
    {
        auto cam_valid = torch::ones({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());
        auto masks     = dataloader->getMasks(cam_valid);
        auto images    = dataloader->getImages(dataloader->getLastAvailableFrame(), cam_valid);

        size_t message_size = sizeof(rift::protocol::ProtocolHeader) + 3 * sizeof(uint32_t);

        for(auto& mask: masks)
        {
            message_size += mask.element_size() * mask.numel() + sizeof(uint32_t);
        }

        for(auto& img: images)
        {
            message_size += img.size() + sizeof(uint32_t);
        }

        std::vector<uint8_t> response(message_size);
        auto header = rift::protocol::ProtocolHeader(rift::protocol::MessageTask::MASKS_IMAGES);
        std::memcpy(response.data(), &header, sizeof(rift::protocol::ProtocolHeader));
        uint32_t offset = sizeof(rift::protocol::ProtocolHeader);

        atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, (uint32_t)masks.size());
        atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, dataloader->width());
        atcg::NetworkUtils::writeInt<uint32_t>(response.data(), offset, dataloader->height());

        for(auto& mask: masks)
        {
            atcg::NetworkUtils::writeBuffer(response.data(),
                                            offset,
                                            (uint8_t*)mask.cpu().data_ptr(),
                                            mask.numel() * mask.element_size());
        }

        for(auto& img: images)
        {
            atcg::NetworkUtils::writeBuffer(response.data(), offset, img.data(), img.size());
        }

        return response;
    }

    // Currently runs on the network thread
    void onReceive(std::vector<uint8_t>& data, const uint64_t client_id)
    {
        uint32_t offset = 0;
        rift::protocol::ProtocolHeader header =
            atcg::NetworkUtils::readStruct<rift::protocol::ProtocolHeader>(data.data(), offset);

        if(!header.version.isCompatible())
        {
            ATCG_ERROR("Protocols not compatible. This version of the protocol is {}.{}. Received message had version "
                       "{}.{}",
                       RIFTCAST_PROTOCOL_VERSION_MAJOR,
                       RIFTCAST_PROTOCOL_VERSION_MINOR,
                       header.version.major,
                       header.version.minor);
            return;
        }

        std::vector<uint8_t> response;
        switch(header.task)
        {
            case rift::protocol::MessageTask::CONNECT_CLIENT:
            {
                process_connect(client_id);
                return;    // Dont send response
            }
            break;
            case rift::protocol::MessageTask::DISCONNECT_CLIENT:
            {
                process_disconnect(client_id);
                return;    // Dont send response
            }
            break;
            case rift::protocol::MessageTask::REQUEST_IMAGE:
            {
                response = process_render(data.data(), offset, client_id);
            }
            break;
            case rift::protocol::MessageTask::MASKS:
            {
                response = process_masks();
            }
            break;
            case rift::protocol::MessageTask::MASKS_IMAGES:
            {
                response = process_masks_images();
            }
            break;
            case rift::protocol::MessageTask::CAMERA_DATA:
            {
                response = process_cameras();
            }
            break;
        }

        if(!response.empty())
        {
            server.sendToClient(response, client_id);
        }
    }

    void onReceiveQueue(std::vector<uint8_t>& data, const uint64_t client_id)
    {
        std::lock_guard guard(queue_mutex);

        queue.push(std::make_unique<DataPacket>(data, client_id));
    }

    void onConnect(const uint64_t client_id)
    {
        // When we get a new connection, we want to create a new client state but this involves OpenGL functions -> has
        // to be done on main thread
        std::lock_guard guard(queue_mutex);

        auto message = rift::protocol::createConnectionMessage();
        queue.push(std::make_unique<DataPacket>(message, client_id));
    }

    void onDisconnect(const uint64_t client_id)
    {
        // Also have to destroy client state on main thread
        std::lock_guard guard(queue_mutex);

        auto message = rift::protocol::createDisconnectionMessage();
        queue.push(std::make_unique<DataPacket>(message, client_id));
    }

    void show_statistics()
    {
        std::cout << "\n\n\n\n\n\n\n";    // "Allocate" six lines
        while(running)
        {
            {
                std::lock_guard guard(statistic_mutex);
                std::cout << "\033[F\033[F\033[F\033[F\033[F"
                          << "\r\033[K\rStatistics:\n"
                          << "\r\033[K\rConnected clients: " << client_states.size() << "\n"
                          << "\r\033[K" << rendering_statistic << "\r\033[K" << visual_hull_statistic << "\r\033[K"
                          << inpaint_statistic << std::flush;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void cli()
    {
        std::thread show_statistics_thread(std::bind(&RIFTCastServer::show_statistics, this));

        while(running)
        {
            std::string input;
            std::cin >> input;

            if(input == "exit")
            {
                running = false;
            }
        }

        if(show_statistics_thread.joinable()) show_statistics_thread.join();
    }

    void visual_hull()
    {
        int visual_hull_device_id = dataloader->reconstruction_gpu();

        at::cuda::CUDAGuard device_guard(visual_hull_device_id);
        auto visual_hull_stream = at::cuda::getStreamFromPool();
        at::cuda::CUDAStreamGuard stream_guard(visual_hull_stream);

        auto cam_valid = torch::ones({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());

        auto geometry_module = atcg::make_ref<rift::GeometryModule>();
        geometry_module->init(visual_hull_device_id, dataloader);

        float delta_time = 1.0f / 60.0f;

        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream base;

        base << "bin/log/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        auto uuid = atcg::UUID();

        auto reconstruction_logger =
            rift::BenchmarkLogger("reconstruction_" + std::to_string(uuid), base.str() + "_reconstruction.txt");

        while(running)
        {
            atcg::Timer timer;
            auto reconstruction = geometry_module->compute_geometry(model, cam_valid);
            {
                std::unique_lock vh_guard(visual_hull_mutex);
                output_vertices   = reconstruction.vertices;
                output_faces      = reconstruction.faces;
                output_normals    = reconstruction.normals;
                output_primitives = reconstruction.visible_primitives;
                current_frame     = dataloader->getLastAvailableFrame();
            }

            reconstruction_logger.logSample(timer.elapsedMillis());
            delta_time = timer.elapsedSeconds();

            {
                std::lock_guard guard(statistic_mutex);
                visual_hull_statistic.addSample(timer.elapsedMillis());
            }
        }

        geometry_module.reset();
    }

    void render(ClientState* state)
    {
        at::cuda::CUDAGuard device_guard(dataloader->renderer_gpu());

        auto render_stream = at::cuda::getStreamFromPool();
        at::cuda::CUDAStreamGuard stream_guard(render_stream);

        atcg::ref_ptr<atcg::PerspectiveCamera> camera = atcg::make_ref<atcg::PerspectiveCamera>();

        auto render_module = atcg::make_ref<rift::RenderModule>();
        render_module->init(dataloader->renderer_gpu(), dataloader, atcg::JPEGBackend::HARDWARE);

        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream base;

        base << "bin/log/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        auto uuid = atcg::UUID();

        auto render_logger = rift::BenchmarkLogger("render_" + std::to_string(uuid), base.str() + "_render.txt");
        atcg::Timer recompile_timer;
        atcg::Worker worker;
        worker.start();
        while(true)
        {
            if(!state->running)
            {
                break;
            }

            if(!state->start_rendering)
            {
                continue;
            }

            state->start_rendering = false;

            atcg::Timer timer;

            // 2. Get camera parameters
            uint32_t width  = state->render_input_width;
            uint32_t height = state->render_input_height;

            glm::mat4 view       = state->render_input_view;
            glm::mat4 projection = state->render_input_projection;
            camera->setView(view);
            camera->setProjection(projection);

            rift::GeometryReconstruction reconstruction;
            {
                std::shared_lock lock(visual_hull_mutex);
                reconstruction.current_frame = current_frame;
                if(output_vertices.numel() > 0 && output_faces.numel() > 0)
                {
                    // visual_hull_done_event.block(render_stream);
                    reconstruction.vertices           = output_vertices.to("cuda:0");
                    reconstruction.faces              = output_faces.to("cuda:0");
                    reconstruction.normals            = output_normals.to("cuda:0");
                    reconstruction.visible_primitives = output_primitives.to("cuda:0");
                    // render_stream.synchronize();
                }
            }

            render_module->updateState(reconstruction, camera, width, height);

            // SMPL-X reconstruction
            {
                // Build list of selected camera IDs in order
                torch::Tensor cam_valid = render_module->getChosenCameraIndices();
                auto cam_valid_cpu = cam_valid.to(torch::kCPU);
                auto cam_valid_accessor = cam_valid_cpu.accessor<int,1>();

                std::vector<int> selected_cam_indices;
                for (int i = 0; i < cam_valid_cpu.size(0); ++i)
                {
                    if (cam_valid_accessor[i] == 1) 
                    {
                        selected_cam_indices.push_back(i);
                    }
                }

                // Get images (this returns them in the order they were selected)
                torch::Tensor selected_images = render_module->getSelectedCamerasImages();
                
                // Get actual camera IDs from dataloader
                const auto& cameras = dataloader->getCameras();
                
                // Build image tensors map (cam_id -> tensor)
                std::map<std::string, torch::Tensor> image_tensors;
                std::vector<std::string> selected_cam_ids_str;
                
                for (int idx = 0; idx < selected_cam_indices.size(); ++idx)
                {
                    int cam_index = selected_cam_indices[idx];
                    std::string cam_id = std::to_string(cameras[cam_index].id);  // Get actual camera ID/name
                    torch::Tensor img_tensor = selected_images[idx];
                    
                    image_tensors[cam_id] = img_tensor;
                    selected_cam_ids_str.push_back(cam_id);
                }

                // Call SMPL-X reconstruction via ZMQ
                try
                {
                    atcg::Timer smplx_timer;
                    smplx_client.process_frame(image_tensors, selected_cam_ids_str, smplx_result);
                    float smplx_time = smplx_timer.elapsedMillis();
                    // std::cout << "SMPL-X processing time: " << smplx_time << " ms" << std::endl;

                    if (smplx_result.frame_index == -1)
                    {
                        std::cout << "SMPL-X reconstruction failed." << std::endl;
                    }
                    else
                    {
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "SMPL-X reconstruction exception: " << e.what() << std::endl;
                }
            }

            auto framebuffer = render_module->renderFrame(camera);

            // 5. Encode image
            auto img_data        = framebuffer->getColorAttachement(0)->getData(atcg::GPU);
            auto visibility_data = framebuffer->getColorAttachement(2)->getData(atcg::GPU);
            auto depth_data      = framebuffer->getColorAttachement(3)->getData(atcg::GPU);

            auto mask = torch::where(visibility_data == 0, 1.0f, 0.0f);


            depth_data = rift::quantize(depth_data, camera->getNear(), camera->getFar()).cpu();
            {
                // Send output to inpainting thread
                std::lock_guard guard(state->render_mutex);
                state->render_output_img                 = img_data;
                state->render_output_depth               = depth_data;
                state->render_output_mask                = mask;
                state->render_output_inv_view_projection = glm::inverse(projection * view);
                state->render_output_frame_idx           = current_frame;
                render_stream.synchronize();

                state->rendering_done = true;
                state->start_inpaint  = true;
            }

            render_logger.logSample(timer.elapsedMillis());

            if(recompile_timer.elapsedSeconds() >= 1.0f)
            {
                recompile_timer.reset();
                render_module->getShaderManager()->onUpdate();
            }

            {
                std::lock_guard guard(statistic_mutex);
                rendering_statistic.addSample(timer.elapsedMillis());
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        render_module.reset();
    }

    void inpaint(ClientState* state)
    {
        auto inpainting_stream = at::cuda::getStreamFromPool();
        at::cuda::CUDAStreamGuard stream_guard(inpainting_stream);

        auto encoder = atcg::make_ref<atcg::JPEGEncoder>(true, atcg::JPEGBackend::HARDWARE);

        auto inpainting_module = atcg::make_ref<rift::InpaintingModule>(dataloader->inpainting_path());

        auto t  = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream base;

        base << "bin/log/" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        auto uuid = atcg::UUID();

        auto inpaint_logger = rift::BenchmarkLogger("inpaint_" + std::to_string(uuid), base.str() + "_inpaint.txt");

        while(true)
        {
            if(!state->running)
            {
                break;
            }

            if(!state->start_inpaint)
            {
                continue;
            }

            state->start_inpaint = true;

            atcg::Timer timer;

            torch::Tensor img_tensor;
            torch::Tensor mask_tensor;
            torch::Tensor depth_tensor;
            glm::mat4 inv_view_projection;

            uint32_t local_frame_idx = 0;
            {
                // Get data from render thread
                std::lock_guard guard(state->render_mutex);

                img_tensor          = state->render_output_img.clone();
                mask_tensor         = state->render_output_mask.clone();
                depth_tensor        = state->render_output_depth.clone();
                inv_view_projection = state->render_output_inv_view_projection;
                local_frame_idx     = state->render_output_frame_idx;
            }
            auto img_data   = img_tensor;
            auto depth_data = depth_tensor;

            if(dataloader->enable_inpainting())
            {
                auto pred_img = inpainting_module->inpaint(img_tensor, mask_tensor);
                img_data      = pred_img;
            }

            // Compress results
            size_t compressedSize        = ZSTD_compressBound(depth_data.numel() * depth_data.element_size());
            torch::Tensor compressedData = torch::empty({(int)compressedSize}, atcg::TensorOptions::uint8HostOptions());

            compressedSize = ZSTD_compress(compressedData.data_ptr(),
                                           compressedSize,
                                           depth_data.data_ptr(),
                                           depth_data.numel() * depth_data.element_size(),
                                           1);

            compressedData.resize_(compressedSize);

            if(ZSTD_isError(compressedSize))
            {
                ATCG_ERROR(ZSTD_getErrorName(compressedSize));
            }

            auto encoded = encoder->compress(img_data).contiguous();

            inpaint_logger.logSample(timer.elapsedMillis());

            {
                std::lock_guard guard(state->inpaint_mutex);
                static std::array<float, 178> smplx_params{};

                {
                    size_t offset = 0;

                    // Global orient (3 floats)
                    std::copy(smplx_result.global_orient.begin(), smplx_result.global_orient.end(), smplx_params.begin() + offset);
                    offset += 3;

                    // Body pose (63 floats)
                    std::copy(smplx_result.body_pose.begin(), smplx_result.body_pose.end(), smplx_params.begin() + offset);
                    offset += 63;

                    // Left hand pose (45 floats)
                    std::copy(smplx_result.left_hand_pose.begin(), smplx_result.left_hand_pose.end(), smplx_params.begin() + offset);
                    offset += 45;

                    // Right hand pose (45 floats)
                    std::copy(smplx_result.right_hand_pose.begin(), smplx_result.right_hand_pose.end(), smplx_params.begin() + offset);
                    offset += 45;

                    // Jaw pose (3 floats)
                    std::copy(smplx_result.jaw_pose.begin(), smplx_result.jaw_pose.end(), smplx_params.begin() + offset);
                    offset += 3;

                    offset = 165;
                    std::copy(smplx_result.betas.begin(), smplx_result.betas.end(), smplx_params.begin() + offset);
                    offset += 10;

                    std::copy(smplx_result.transl.begin(), smplx_result.transl.end(), smplx_params.begin() + offset);
                }

                state->current_response =
                    std::move(rift::protocol::createUpdateMessage(
                        inv_view_projection, 
                        encoded, 
                        compressedData,
                        local_frame_idx,
                        smplx_params));
                state->inpaint_done = true;
            }

            {
                std::lock_guard guard(statistic_mutex);
                inpaint_statistic.addSample(timer.elapsedMillis());
            }


            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    virtual void run() override
    {
        auto& cmd_args = atcg::getCommandLineArguments();

        if(cmd_args.size() != 2)
        {
            ATCG_ERROR("Usage: ./bin/RIFTCast_server <config_path>");
            return;
        }

        auto header = rift::IO::readDatasetHeader(cmd_args[1]);

        // Per default: no context should be active
        main_context = atcg::Application::get()->getWindow()->getContext();
        atcg::Renderer::toggleCulling(false);

        main_context->deactivate();

        auto transform = atcg::TransformComponent(header.volume_position, glm::vec3(header.volume_scale), glm::vec3(0));
        model          = transform.getModel();

        dataloader = rift::createDatasetImporter(header);

        rendering_statistic   = atcg::Statistic<float>("Rendering");
        visual_hull_statistic = atcg::Statistic<float>("Visual Hull");
        inpaint_statistic     = atcg::Statistic<float>("Inpainting");

        server.setOnReceiveCallback(
            std::bind(&RIFTCastServer::onReceiveQueue, this, std::placeholders::_1, std::placeholders::_2));

        server.setOnConnectCallback(std::bind(&RIFTCastServer::onConnect, this, std::placeholders::_1));
        server.setOnDisconnectCallback(std::bind(&RIFTCastServer::onDisconnect, this, std::placeholders::_1));
        server.start(header.server_ip, header.server_port);
        running = true;

        std::thread cli_thread(&RIFTCastServer::cli, this);
        std::thread visual_hull_thread(&RIFTCastServer::visual_hull, this);

        // Init SMPL-X reconstruction
        {
            // const std::string default_vci_dir =
            //     "/data/spindler/Code/VisualComputingGroup/RIFTCast/res/bonn_tele_bench_001";
            const std::string default_vci_dir =
                "/data/spindler/Code/VisualComputingGroup/RIFTCast/res/bonn_tele_bench_001";
            smplx_client.initialize(default_vci_dir);
        }

        // Check for data to send
        while(running)
        {
            std::unique_ptr<DataPacket> data = nullptr;
            {
                std::lock_guard guard(queue_mutex);

                if(!queue.empty())
                {
                    data = std::move(queue.front());
                    queue.pop();
                }
            }

            if(data)
            {
                onReceive(data->data, data->client_id);
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        if(cli_thread.joinable()) cli_thread.join();
        if(visual_hull_thread.joinable()) visual_hull_thread.join();

        server.stop();

        main_context->makeCurrent();
    }

private:
    std::atomic_bool running = false;
    atcg::TCPServer server;

    atcg::ref_ptr<atcg::Context> main_context;

    std::mutex queue_mutex;
    std::queue<std::unique_ptr<DataPacket>> queue;

    std::shared_mutex visual_hull_mutex;
    torch::Tensor output_vertices, output_faces, output_normals, output_primitives;
    int current_frame = 0;

    std::unordered_map<uint64_t, atcg::ref_ptr<ClientState>> client_states;

    atcg::ref_ptr<rift::DatasetImporter> dataloader;
    int level = 9;
    glm::mat4 model;

    atcg::Statistic<float> rendering_statistic;
    atcg::Statistic<float> visual_hull_statistic;
    atcg::Statistic<float> inpaint_statistic;
    std::mutex statistic_mutex;

    // SMPL-X reconstruction
    SMPLXClient smplx_client = SMPLXClient("tcp://localhost:5555");
    SMPLXResult smplx_result;
};

atcg::Application* atcg::createApplication()
{
    // vci::interceptor::setup_logger(spdlog::level::info);    //

    atcg::WindowProps props;
    props.vsync  = false;
    props.hidden = true;
    return new RIFTCastServer(props);
}
