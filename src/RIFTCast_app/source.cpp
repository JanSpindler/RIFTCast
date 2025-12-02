#define BUILD_VR true

#include <iostream>

#include <Core/EntryPoint.h>
#include <ATCG.h>

#include <algorithm>

#include <random>

#include <riftcast/DatasetImporter.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include <portable-file-dialogs.h>

#include <thread>
#include <mutex>

#include <riftcast/GeometryModule.h>
#include <riftcast/RenderModule.h>
#include <riftcast/riftcastkernels.h>

#ifndef ATCG_HEADLESS
    #include <implot.h>
#endif

#include <zmq.hpp>
#include <json.hpp>

using json = nlohmann::json;

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

    json process_frame(
        const std::map<std::string, torch::Tensor>& image_tensors,
        const std::vector<std::string>& selected_cam_ids)
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
        
        // Receive response
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);
        
        std::string reply_str(static_cast<char*>(reply.data()), reply.size());
        return json::parse(reply_str);
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

class RIFTCastLayer : public atcg::Layer
{
public:
    RIFTCastLayer(const std::string& name) : atcg::Layer(name) {}

    void destroy()
    {
        running = false;
        if(visual_hull_thread.joinable()) visual_hull_thread.join();
        if(render_thread.joinable()) render_thread.join();

        // geometry_module.reset();
        // render_module.reset();
        atcg::Renderer::use();
    }

    void visual_hull()
    {
        int visual_hull_device_id = 0;

        at::cuda::CUDAGuard device_guard(visual_hull_device_id);
        auto cam_valid = torch::ones({dataloader->num_cameras()}, atcg::TensorOptions::int32DeviceOptions());

        auto geometry_module = atcg::make_ref<rift::GeometryModule>();
        geometry_module->init(visual_hull_device_id, dataloader);

        float delta_time = 1.0f / 60.0f;
        while(running)
        {
            atcg::Timer timer;
            auto reconstruction = geometry_module->compute_geometry(model, cam_valid);
            {
                std::lock_guard vh_guard(visual_hull_mutex);
                output_vertices   = reconstruction.vertices;
                output_faces      = reconstruction.faces;
                output_normals    = reconstruction.normals;
                output_primitives = reconstruction.visible_primitives;
                current_frame     = dataloader->getLastAvailableFrame();
            }


            delta_time = timer.elapsedSeconds();

            {
                std::lock_guard guard(statistic_mutex);
                current_reconstruction_time = timer.elapsedMillis();
            }
        }

        geometry_module.reset();
    }

    atcg::ref_ptr<rift::RenderModule> render_module;
    void render()
    {
        render_module = atcg::make_ref<rift::RenderModule>();
        render_module->init(0, dataloader, atcg::JPEGBackend::SOFTWARE);

        atcg::CameraExtrinsics extrinsics;
        atcg::CameraIntrinsics intrinsics;
        atcg::ref_ptr<atcg::PerspectiveCamera> camera = atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics);

        atcg::Timer recompile_timer;

        while(true)
        {
            if(!running)
            {
                break;
            }

            if(!start_rendering)
            {
                continue;
            }

            start_rendering = false;

            atcg::Timer timer;

            // 2. Get camera parameters
            uint32_t width  = render_input_width;
            uint32_t height = render_input_height;

            glm::mat4 view       = render_input_view;
            glm::mat4 projection = render_input_projection;
            camera->setView(view);
            camera->setProjection(projection);

            rift::GeometryReconstruction reconstruction;
            {
                std::lock_guard lock(visual_hull_mutex);
                reconstruction.current_frame = current_frame;
                if(output_vertices.numel() > 0 && output_faces.numel() > 0)
                {
                    reconstruction.vertices           = output_vertices.clone();
                    reconstruction.faces              = output_faces.clone();
                    reconstruction.normals            = output_normals.clone();
                    reconstruction.visible_primitives = output_primitives.clone();
                    torch::cuda::synchronize();    // Memcpys are async
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
                    json result = smplx_client.process_frame(image_tensors, selected_cam_ids_str);
                    
                    if (result.contains("status"))
                    {
                        if (result["status"] == "success")
                        {
                            std::cout << "SMPL-X reconstruction successful" << std::endl;
                            
                            // Extract SMPL-X parameters if needed
                            // auto data = result["data"];
                            // auto global_orient = data["global_orient"]; // etc.
                        }
                        else if (result["status"] == "no_detection")
                        {
                            std::cout << "No person detected in frame" << std::endl;
                        }
                        else if (result["status"] == "error")
                        {
                            std::cerr << "SMPL-X error: " << result["error"] << std::endl;
                        }
                    }
                }
                catch (const std::exception& e)
                {
                    std::cerr << "SMPL-X reconstruction exception: " << e.what() << std::endl;
                }
                
                // Build images dict
                // py::dict images_dict;
                // for (int idx = 0; idx < selected_cam_id_list.size(); ++idx)
                // {
                //     int cam_id = selected_cam_id_list[idx];
                //     torch::Tensor img_tensor = selected_images[idx];  // ✓ This is correct
                //     images_dict[py::str(std::to_string(cam_id))] = tensor_to_numpy(img_tensor);
                // }

                // Build Python list for selected_cam_ids
                // py::list selected_cam_ids;
                // for (int cam_id : selected_cam_id_list)
                // {
                //     selected_cam_ids.append(cam_id);
                // }

                // py::object result_obj = smplx_reconstructor.attr("process_frame")(images_dict, selected_cam_ids);
                
                // if (!result_obj.is_none())
                // {
                //     py::dict result = result_obj.cast<py::dict>();
                    
                //     auto global_orient_np = result["global_orient"].cast<py::array_t<float>>();
                //     auto body_pose_np = result["body_pose"].cast<py::array_t<float>>();
                //     auto betas_np = result["betas"].cast<py::array_t<float>>();
                //     auto transl_np = result["transl"].cast<py::array_t<float>>();
                //     auto left_hand_pose_np = result["left_hand_pose"].cast<py::array_t<float>>();
                //     auto right_hand_pose_np = result["right_hand_pose"].cast<py::array_t<float>>();
                //     auto jaw_pose_np = result["jaw_pose"].cast<py::array_t<float>>();
                //     auto expression_np = result["expression"].cast<py::array_t<float>>();
                    
                //     // Use the SMPL-X parameters...
                // }
            }

            auto framebuffer = render_module->renderFrame(camera);

            // 5. Encode image
            auto img_data    = framebuffer->getColorAttachement(0)->getData(atcg::GPU);
            auto depth_data  = framebuffer->getColorAttachement(3)->getData(atcg::GPU);
            auto normal_data = framebuffer->getColorAttachement(4)->getData(atcg::GPU);

            {
                // Send output to inpainting thread
                std::lock_guard guard(render_mutex);
                mesh_frame_idx                    = reconstruction.current_frame;
                render_output_img                 = img_data;
                render_output_depth               = depth_data;
                render_output_normals             = normal_data;
                render_output_inv_view_projection = glm::inverse(projection * view);

                rendering_done = true;
            }

            {
                std::lock_guard guard(statistic_mutex);
                current_mapping_time = timer.elapsedMillis();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            if(recompile_timer.elapsedSeconds() >= 1.0f)
            {
                render_module->getShaderManager()->onUpdate();
                recompile_timer.reset();
            }
        }

        render_module.reset();
    }

    void updateReconstruction()
    {
        atcg::Timer timer;

        torch::Tensor output_img;
        torch::Tensor output_depth;
        torch::Tensor output_normals;
        glm::mat4 inv_view_projection;

        uint32_t mesh_frame_idx_local = 0;

        if(rendering_done)
        {
            rendering_done = false;
            // 1. Update input for rendering thread
            render_input_width =
                (BUILD_VR && atcg::VR::isVRAvailable()) ? atcg::VR::width() : atcg::Renderer::getFramebuffer()->width();
            render_input_height     = (BUILD_VR && atcg::VR::isVRAvailable()) ? atcg::VR::height()
                                                                              : atcg::Renderer::getFramebuffer()->height();
            render_input_view       = camera_controller->getCamera()->getView();
            render_input_projection = camera_controller->getCamera()->getProjection();

            mesh_frame_idx_local = mesh_frame_idx;

            // 2. Get pointcloud data
            if(render_output_img.numel() > 0)    // Should only happen for first frame
            {
                output_img          = render_output_img.clone();
                output_depth        = render_output_depth.clone();
                output_normals      = render_output_normals.clone();
                inv_view_projection = render_output_inv_view_projection;
                torch::cuda::synchronize();
            }


            // 3. Start rendering thread again
            start_rendering = true;
        }

        if(output_img.numel() <= 0) return;

        // Update point cloud
        auto [vertices, colors, normals] =
            rift::unprojectVertices(output_img, inv_view_projection, output_depth, output_normals);
        if(vertices.numel() > 0)
        {
            // Load SMPL-X mesh and set current frame
            // if (smplx_graphs.size() <= mesh_frame_idx_local) 
            // {
            //     smplx_graphs.resize(mesh_frame_idx_local + 1); 
            //     smplx_graphs[mesh_frame_idx_local] = nullptr;
            // }
            // if (smplx_graphs[mesh_frame_idx_local] == nullptr)
            // {
            //     for (size_t frame_idx = 0; frame_idx < smplx_graphs.size(); ++frame_idx) 
            //     {
            //         if (smplx_graphs[frame_idx] == nullptr) 
            //         {
            //             const std::string mesh_path = "./res/meshes/smplest_x_mesh_" + std::to_string(frame_idx) + ".obj";
            //             std::cout << "Loading mesh: " << mesh_path << std::endl;
            //             smplx_graphs[frame_idx] = atcg::IO::read_mesh(mesh_path);
            //         }
            //     }
            // }
            // auto& geometry = mesh_entity.getComponent<atcg::GeometryComponent>();
            // geometry.graph = smplx_graphs[mesh_frame_idx_local];

            // Update SMPL-X translation for moving out of origin
            auto& transform = mesh_entity.getComponent<atcg::TransformComponent>();
            transform.setPosition(glm::vec3(1.0f, 0.0f, 1.0f) * static_cast<float>(mesh_frame_idx_local) * 0.005f + glm::vec3(-0.35f, 0.0f, -0.25f));

            // Point cloud
            pointcloud->resizeVertices(vertices.size(0));
            pointcloud->getDevicePositions().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, vertices);
            pointcloud->getDeviceColors().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, colors);
            pointcloud->getDeviceNormals().index_put_({torch::indexing::Slice(), torch::indexing::Slice()}, normals);
            pointcloud->unmapDeviceVertexPointer();
        }

        current_synchronizing_time = timer.elapsedMillis();
    }

    // This is run at the start of the program
    virtual void onAttach() override
    {
        // Init smplx client
        {
            smplx_client.initialize("/data/jspindle/vci_data");

            // // Aquire GIL
            // py::gil_scoped_acquire acquire;

            // // Add scrypts folder to python
            // py::module_ sys = py::module_::import("sys");
            // sys.attr("path").attr("append")("SMPLest-X");

            // // Import script
            // py::module_ script = py::module_::import("main.vci_server");

            // // Create reconstructor
            // smplx_reconstructor = script.attr("create_reconstructor")();

            // // Init session
            // std::string vci_dir = "/data/jspindle/vci_data";
            // smplx_reconstructor.attr("initialize_session")(vci_dir);
        }        

        atcg::Application::get()->enableDockSpace(true);
        atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));
        atcg::Renderer::toggleCulling(false);
        atcg::Renderer::toggleMSAA(false);

        const auto& window = atcg::Application::get()->getWindow();
        scene              = atcg::make_ref<atcg::Scene>();

        {
            auto reconstruction = scene->createEntity("Reconstruction");
            reconstruction.addComponent<atcg::TransformComponent>();
            pointcloud = atcg::Graph::createPointCloud();
            reconstruction.addComponent<atcg::GeometryComponent>(pointcloud);
            auto& renderer      = reconstruction.addComponent<atcg::PointRenderComponent>();
            renderer.point_size = 3;
            renderer.shader     = atcg::ShaderManager::getShader("flat");
        }

        uint32_t width, height;
        if(BUILD_VR && atcg::VR::isVRAvailable())
        {
            atcg::VR::setNear(0.01f);
            atcg::VR::setFar(10.0f);
            width              = atcg::VR::width();
            height             = atcg::VR::height();
            float aspect_ratio = (float)width / (float)height;
            atcg::CameraIntrinsics instrinsics_left(atcg::VR::getProjection(atcg::VRSystem::Eye::LEFT));
            atcg::CameraIntrinsics instrinsics_right(atcg::VR::getProjection(atcg::VRSystem::Eye::RIGHT));
            instrinsics_left.setAspectRatio(aspect_ratio);
            instrinsics_right.setAspectRatio(aspect_ratio);

            atcg::CameraExtrinsics extrinsics_left(glm::inverse(atcg::VR::getInverseView(atcg::VRSystem::Eye::LEFT)));
            atcg::CameraExtrinsics extrinsics_right(glm::inverse(atcg::VR::getInverseView(atcg::VRSystem::Eye::RIGHT)));
            camera_controller = atcg::make_ref<atcg::VRController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics_left, instrinsics_left),
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics_right, instrinsics_right));
            atcg::VR::initControllerMeshes(scene);
        }
        else
        {
            const auto& window = atcg::Application::get()->getWindow();
            width              = window->getWidth();
            height             = window->getHeight();
            float aspect_ratio = (float)width / (float)height;
            atcg::CameraExtrinsics extrinsics;
            atcg::CameraIntrinsics intrinsics;
            intrinsics.setAspectRatio(aspect_ratio);
            camera_controller = atcg::make_ref<atcg::FirstPersonController>(
                atcg::make_ref<atcg::PerspectiveCamera>(extrinsics, intrinsics));
        }

        // camera_controller->getCamera()->setPosition(glm::vec3(3.0f, 1.0f, 0.0f));
        // camera_controller->getCamera()->setLookAt(glm::vec3(0.0f, 1.0f, 0.0f));

        auto skybox = atcg::IO::imread("res/skybox_vci.hdr");
        scene->setSkybox(skybox);

        {
            mesh_entity     = scene->createEntity("SMPL-X Mesh");
            auto& transform = mesh_entity.addComponent<atcg::TransformComponent>();
            mesh_entity.addComponent<atcg::GeometryComponent>(nullptr);
            // auto& renderer = mesh_entity.addComponent<atcg::MeshRenderComponent>();
        }

        auto f = pfd::open_file("Choose scene meta file", pfd::path::home(), {"Json", "*.json"}, pfd::opt::none);
        std::string meta_file = f.result()[0];

        rift::DatasetHeader header = rift::IO::readDatasetHeader(meta_file);
        dataloader                 = rift::createDatasetImporter(header);

        atcg::TransformComponent transform;
        transform.setScale(glm::vec3(header.volume_scale));
        transform.setPosition(header.volume_position);
        model = transform.getModel();

        {
            auto volume = scene->createEntity("Volume");
            volume.addComponent<atcg::TransformComponent>(transform);
            std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1.0f, -1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(-1.0f, -1.0f, 1.0f)),
                                                  atcg::Vertex(glm::vec3(-1.0f, 1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(-1.0f, 1.0f, 1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, -1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, -1.0f, 1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, 1.0f, -1.0f)),
                                                  atcg::Vertex(glm::vec3(1.0f, 1.0f, 1.0f))};

            std::vector<atcg::Edge> edges = {atcg::Edge {glm::vec2(0, 1), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(0, 2), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(0, 4), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(1, 5), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(1, 3), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(2, 6), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(2, 3), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(3, 7), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(6, 4), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(6, 7), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(4, 5), glm::vec3(1), 1.0f},
                                             atcg::Edge {glm::vec2(5, 7), glm::vec3(1), 1.0f}};

            volume.addComponent<atcg::GeometryComponent>(atcg::Graph::createGraph(vertices, edges));
            volume.addComponent<atcg::EdgeRenderComponent>();
        }

        const auto& cameras = dataloader->getCameras();

        for(int i = 0; i < cameras.size(); ++i)
        {
            atcg::ref_ptr<atcg::PerspectiveCamera> cam = cameras[i].cam;
            auto entity                                = scene->createEntity(cameras[i].name);
            auto& transform                            = entity.addComponent<atcg::TransformComponent>();

            transform.setModel(glm::inverse(cam->getView()) *
                               glm::scale(glm::vec3(1.0f, 1.0f, glm::tan(glm::radians(cam->getFOV()) / 2.0f))));
            auto& component        = entity.addComponent<atcg::CameraComponent>(cam);
            component.render_scale = 0.2f;
            component.width        = cameras[i].width;
            component.height       = cameras[i].height;
        }

        scene->setCamera(camera_controller->getCamera());

        panel = atcg::SceneHierarchyPanel(scene);

        running            = true;
        visual_hull_thread = std::thread(&RIFTCastLayer::visual_hull, this);
        render_thread      = std::thread(&RIFTCastLayer::render, this);
    }

    // This gets called each frame
    virtual void onUpdate(float delta_time) override
    {
        if(!camera_ride)
        {
            camera_controller->onUpdate(delta_time);
        }

        updateReconstruction();

        atcg::Renderer::setClearColor(glm::vec4(1));
        atcg::Renderer::clear();

        if(BUILD_VR && atcg::VR::isVRAvailable())
        {
            atcg::ref_ptr<atcg::VRController> controller =
                std::dynamic_pointer_cast<atcg::VRController>(camera_controller);

            if(controller->inMovement())
            {
                atcg::VR::setMovementLine(controller->getControllerPosition(), controller->getControllerIntersection());
            }

            auto [t_left, t_right] = atcg::VR::getRenderTargets();

            t_left->use();
            atcg::Renderer::setViewport(0, 0, atcg::VR::width(), atcg::VR::height());

            atcg::Renderer::clear();

            controller->getCameraLeft()->setFar(1000.0f);
            scene->draw(controller->getCameraLeft());
            controller->getCameraLeft()->setFar(10.0f);

            atcg::Renderer::drawCameras(scene, controller->getCameraLeft());

            atcg::Renderer::drawCADGrid(controller->getCameraLeft());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraLeft());
            }

            t_right->use();

            atcg::Renderer::clear();

            controller->getCameraRight()->setFar(1000.0f);
            scene->draw(controller->getCameraRight());
            controller->getCameraRight()->setFar(10.0f);

            atcg::Renderer::drawCameras(scene, controller->getCameraRight());

            atcg::Renderer::drawCADGrid(controller->getCameraRight());


            // atcg::Renderer::drawCADGrid(controller->getCameraRight());

            if(controller->inMovement())
            {
                atcg::VR::drawMovementLine(controller->getCameraRight());
            }

            atcg::Renderer::useScreenBuffer();
            atcg::Renderer::setDefaultViewport();

            atcg::VR::renderToScreen();
        }
        else
        {
            atcg::Renderer::clear();
            camera_controller->getCamera()->setFar(1000.0f);
            scene->draw(camera_controller->getCamera());
            camera_controller->getCamera()->setFar(10.0f);
            atcg::Renderer::drawCameras(scene, camera_controller->getCamera());

            atcg::Renderer::drawCADGrid(camera_controller->getCamera());
        }


        time_passed += delta_time;

        // if(time_passed > 1.0f)
        {
            std::lock_guard guard(statistic_mutex);
            time_passed         = 0.0f;
            reconstruction_time = current_reconstruction_time;
            mapping_time        = current_mapping_time;
            synchronizing_time  = current_synchronizing_time;
            display_time        = delta_time;
        }

        if(camera_ride)
        {
            static float angle = 0.0f;

            angle += 0.05f * delta_time;

            float c = glm::cos(atcg::Constants::two_pi<float>() * angle);
            float s = glm::sin(atcg::Constants::two_pi<float>() * angle);
            float r = 2.5f;

            camera_controller->getCamera()->setPosition(glm::vec3(r * c, 1.8f, r * s));
            camera_controller->getCamera()->setLookAt(glm::vec3(0.0f, 1.0f, 0.0f));
        }
    }

#ifndef ATCG_HEADLESS
    virtual void onImGuiRender() override
    {
        // return;
        ImGui::BeginMainMenuBar();

        if(ImGui::BeginMenu("File"))
        {
            if(ImGui::MenuItem("Save"))
            {
                atcg::Serializer<atcg::ComponentSerializer> serializer(scene);

                serializer.serialize("../Scene/Scene.yaml");
            }

            if(ImGui::MenuItem("Load"))
            {
                scene = atcg::make_ref<atcg::Scene>();
                atcg::Serializer<atcg::ComponentSerializer> serializer(scene);

                serializer.deserialize("../Scene/Scene.yaml");

                hovered_entity = atcg::Entity();
                panel.selectEntity(hovered_entity);
            }

            if(ImGui::MenuItem("Screenshot"))
            {
                auto t  = std::time(nullptr);
                auto tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "bin/" << "Main" << "_" << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S") << ".png";
                atcg::Renderer::screenshot(scene, camera_controller->getCamera(), 1920, oss.str());
            }

            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();


        ImGui::Begin("RIFTCast");

        static bool use_greedy = true;
        if(ImGui::Checkbox("Selection", &use_greedy))
        {
            render_module->setGreedySelection(use_greedy);
        }

        ImGui::End();

        ImGui::Begin("Reconstruction Statistics");

        ImGui::Text(("Reconstruction Time: " + std::to_string(reconstruction_time) + " ms").c_str());
        ImGui::Text(("Mapping Time:   " + std::to_string(mapping_time) + " ms").c_str());
        ImGui::Text(("Synchronizing Time:  " + std::to_string(synchronizing_time) + " ms").c_str());
        ImGui::Separator();
        ImGui::Text(("Render Thread: " + std::to_string(display_time) + " s / " +
                     std::to_string(1.0f / (display_time)) + " fps")
                        .c_str());
        time_collection.addSample(atcg::Renderer::getFrameCounter());
        fps_collection_mean.addSample(1.0f / (display_time + 1e-5f));
        fps_collection.addSample(fps_collection_mean.mean());

        reconstruction_collection_mean.addSample(1.0f / (reconstruction_time / 1000.0f + 1e-5f));
        reconstruction_collection.addSample(reconstruction_collection_mean.mean());

        synchronizing_collection_mean.addSample(1.0f / (synchronizing_time / 1000.0f + 1e-5f));
        synchronizing_collection.addSample(synchronizing_collection_mean.mean());

        mapping_collection_mean.addSample(1.0f / (mapping_time / 1000.0f + 1e-5f));
        mapping_collection.addSample(mapping_collection_mean.mean());

        ImPlot::SetNextAxisLimits(ImAxis_Y1, 0, 100);
        if(ImPlot::BeginPlot("Runtime"))
        {
            ImPlot::SetupAxes("X", "Y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_None);
            ImPlot::PlotLine("Render Time",
                             time_collection.get(),
                             fps_collection.get(),
                             fps_collection.count(),
                             0,
                             fps_collection.index(),
                             sizeof(float));
            ImPlot::PlotLine("Synchronizing Time",
                             time_collection.get(),
                             synchronizing_collection.get(),
                             synchronizing_collection.count(),
                             0,
                             synchronizing_collection.index(),
                             sizeof(float));
            ImPlot::PlotLine("Reconstruction Time",
                             time_collection.get(),
                             reconstruction_collection.get(),
                             reconstruction_collection.count(),
                             0,
                             reconstruction_collection.index(),
                             sizeof(float));
            ImPlot::PlotLine("Mapping Time",
                             time_collection.get(),
                             mapping_collection.get(),
                             mapping_collection.count(),
                             0,
                             mapping_collection.index(),
                             sizeof(float));
            ImPlot::EndPlot();
        }

        ImGui::End();

        panel.renderPanel();
        hovered_entity = panel.getSelectedEntity();

        atcg::drawGuizmo(scene, hovered_entity, current_operation, camera_controller->getCamera());
    }
#endif

    // This function is evaluated if an event (key, mouse, resize events, etc.) are triggered
    virtual void onEvent(atcg::Event* event) override
    {
        camera_controller->onEvent(event);

        atcg::EventDispatcher dispatcher(event);
        dispatcher.dispatch<atcg::ViewportResizeEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onViewportResized));
#ifndef ATCG_HEADLESS
        dispatcher.dispatch<atcg::MouseMovedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onMouseMoved));
        dispatcher.dispatch<atcg::MouseButtonPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onMousePressed));
        dispatcher.dispatch<atcg::KeyPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onKeyPressed));
        dispatcher.dispatch<atcg::VRButtonPressedEvent>(ATCG_BIND_EVENT_FN(RIFTCastLayer::onVRButtonPressed));
#endif
    }

    bool onViewportResized(atcg::ViewportResizeEvent* event)
    {
        atcg::WindowResizeEvent resize_event(event->getWidth(), event->getHeight());
        camera_controller->onEvent(&resize_event);

        return false;
    }

#ifndef ATCG_HEADLESS
    bool onVRButtonPressed(atcg::VRButtonPressedEvent* event) { return true; }

    bool onKeyPressed(atcg::KeyPressedEvent* event)
    {
        if(event->getKeyCode() == ATCG_KEY_T)
        {
            current_operation = ImGuizmo::OPERATION::TRANSLATE;
        }
        if(event->getKeyCode() == ATCG_KEY_R)
        {
            current_operation = ImGuizmo::OPERATION::ROTATE;
        }
        if(event->getKeyCode() == ATCG_KEY_S)
        {
            current_operation = ImGuizmo::OPERATION::SCALE;
        }

        if(event->getKeyCode() == ATCG_KEY_H)
        {
            hover_mode = !hover_mode;
        }

        if(event->getKeyCode() == ATCG_KEY_K)
        {
            camera_ride = !camera_ride;
        }

        return true;
    }

    bool onMousePressed(atcg::MouseButtonPressedEvent* event)
    {
        if(in_viewport && event->getMouseButton() == ATCG_MOUSE_BUTTON_LEFT && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
            panel.selectEntity(hovered_entity);
        }
        return true;
    }

    bool onMouseMoved(atcg::MouseMovedEvent* event)
    {
        const atcg::Application* app = atcg::Application::get();
        glm::ivec2 offset            = app->getViewportPosition();
        int height                   = app->getViewportSize().y;
        mouse_pos                    = glm::vec2(event->getX() - offset.x, height - (event->getY() - offset.y));

        in_viewport =
            mouse_pos.x >= 0 && mouse_pos.y >= 0 && mouse_pos.y < height && mouse_pos.x < app->getViewportSize().x;


        if(hover_mode && in_viewport && !ImGuizmo::IsOver())
        {
            int id         = atcg::Renderer::getEntityIndex(mouse_pos);
            hovered_entity = id == -1 ? atcg::Entity() : atcg::Entity((entt::entity)id, scene.get());
            panel.selectEntity(hovered_entity);
        }

        return false;
    }
#endif

private:
    float time_passed                                 = 0.0f;
    float reconstruction_time                         = 0.0f;
    float mapping_time                                = 0.0f;
    float synchronizing_time                          = 0.0f;
    float display_time                                = 0.0f;
    atcg::CyclicCollection<float> time_collection     = atcg::CyclicCollection<float>("Time Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> fps_collection      = atcg::CyclicCollection<float>("FPS Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> fps_collection_mean = atcg::CyclicCollection<float>("FPS Collection", 5);
    atcg::CyclicCollection<float> reconstruction_collection =
        atcg::CyclicCollection<float>("Reconstruction Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> reconstruction_collection_mean =
        atcg::CyclicCollection<float>("Reconstruction Collection", 5);
    atcg::CyclicCollection<float> synchronizing_collection =
        atcg::CyclicCollection<float>("Synchronizing Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> synchronizing_collection_mean =
        atcg::CyclicCollection<float>("Synchronizing Collection", 5);
    atcg::CyclicCollection<float> mapping_collection = atcg::CyclicCollection<float>("Mapping Collection", 35 * 60 / 5);
    atcg::CyclicCollection<float> mapping_collection_mean = atcg::CyclicCollection<float>("Mapping Collection", 5);

    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity hovered_entity;

    atcg::Entity mesh_entity;
    std::vector<atcg::ref_ptr<atcg::Graph>> smplx_graphs;
    uint32_t mesh_frame_idx = 0;
    SMPLXClient smplx_client = SMPLXClient("tcp://localhost:5555");

    atcg::ref_ptr<rift::DatasetImporter> dataloader;
    // atcg::ref_ptr<rift::GeometryModule> geometry_module;
    // atcg::ref_ptr<rift::RenderModule> render_module;
    // torch::Tensor cam_valid;
    glm::mat4 model;

    atcg::ref_ptr<atcg::CameraController> camera_controller;

    atcg::SceneHierarchyPanel<atcg::ComponentGUIHandler> panel;

    bool in_viewport = false;

    bool hover_mode  = false;
    bool camera_ride = false;

    glm::vec2 mouse_pos;

    atcg::ref_ptr<atcg::Graph> pointcloud;

    // Visual hull thread
    std::thread visual_hull_thread;
    std::mutex visual_hull_mutex;
    std::atomic_bool running = false;
    torch::Tensor output_vertices, output_faces, output_normals, output_primitives;
    int current_frame = 0;

    // Render thread
    std::thread render_thread;
    std::mutex render_mutex;
    std::atomic_bool start_rendering = false;
    std::atomic_bool rendering_done  = true;
    uint32_t render_input_width, render_input_height;
    glm::mat4 render_input_view, render_input_projection;
    torch::Tensor render_output_img;
    torch::Tensor render_output_depth;
    torch::Tensor render_output_normals;
    glm::mat4 render_output_inv_view_projection;

    // Statistics
    std::mutex statistic_mutex;
    float current_reconstruction_time;
    float current_mapping_time;
    float current_synchronizing_time;

    bool show_render_settings = false;
#ifndef ATCG_HEADLESS
    ImGuizmo::OPERATION current_operation = ImGuizmo::OPERATION::TRANSLATE;
#endif
};

class RIFTCast : public atcg::Application
{
public:
    RIFTCast(const atcg::WindowProps& props) : atcg::Application(props)
    {
        _layer = new RIFTCastLayer("Layer");
        pushLayer(_layer);
    }

    ~RIFTCast() { _layer->destroy(); }

private:
    RIFTCastLayer* _layer;
};

atcg::Application* atcg::createApplication()
{
    atcg::WindowProps props;
    props.vsync = true;
    return new RIFTCast(props);
}