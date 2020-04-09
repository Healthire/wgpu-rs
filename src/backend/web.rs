use std::{
    future::Future,
    ops::Range,
    pin::Pin,
    task::{Context, Poll},
};

use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use wgt::{
    BackendBit, BlendDescriptor, BlendFactor, BlendOperation, BufferAddress, Color,
    ColorStateDescriptor, CompareFunction, CullMode, DepthStencilStateDescriptor, DeviceDescriptor,
    DynamicOffset, Extensions, FrontFace, IndexFormat, InputStepMode, Limits, LoadOp,
    PowerPreference, PrimitiveTopology, RasterizationStateDescriptor, SamplerDescriptor,
    ShaderStage, StencilOperation, StencilStateFaceDescriptor, StoreOp, SwapChainDescriptor,
    TextureComponentType, TextureFormat, TextureViewDimension, VertexAttributeDescriptor,
    VertexFormat,
};

#[derive(Debug, Clone, PartialEq)]
pub struct AdapterId(web_sys::GpuAdapter);
unsafe impl Send for AdapterId {}
unsafe impl Sync for AdapterId {}

#[derive(Debug, Clone, PartialEq)]
pub struct DeviceId(web_sys::GpuDevice);
unsafe impl Send for DeviceId {}
unsafe impl Sync for DeviceId {}

#[derive(Debug, Clone, PartialEq)]
pub struct BufferId(web_sys::GpuBuffer);
unsafe impl Send for BufferId {}
unsafe impl Sync for BufferId {}

#[derive(Debug, Clone, PartialEq)]
pub struct TextureId(web_sys::GpuTexture);
unsafe impl Send for TextureId {}
unsafe impl Sync for TextureId {}

#[derive(Debug, Clone, PartialEq)]
pub struct TextureViewId(web_sys::GpuTextureView);
unsafe impl Send for TextureViewId {}
unsafe impl Sync for TextureViewId {}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerId(web_sys::GpuSampler);
unsafe impl Send for SamplerId {}
unsafe impl Sync for SamplerId {}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceId(web_sys::GpuCanvasContext);
unsafe impl Send for SurfaceId {}
unsafe impl Sync for SurfaceId {}

#[derive(Debug, Clone, PartialEq)]
pub struct SwapChainId(web_sys::GpuSwapChain);
unsafe impl Send for SwapChainId {}
unsafe impl Sync for SwapChainId {}

#[derive(Debug, Clone, PartialEq)]
pub struct BindGroupLayoutId(web_sys::GpuBindGroupLayout);
unsafe impl Send for BindGroupLayoutId {}
unsafe impl Sync for BindGroupLayoutId {}

#[derive(Debug, Clone, PartialEq)]
pub struct BindGroupId(web_sys::GpuBindGroup);
unsafe impl Send for BindGroupId {}
unsafe impl Sync for BindGroupId {}

#[derive(Debug, Clone, PartialEq)]
pub struct ShaderModuleId(web_sys::GpuShaderModule);
unsafe impl Send for ShaderModuleId {}
unsafe impl Sync for ShaderModuleId {}

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineLayoutId(web_sys::GpuPipelineLayout);
unsafe impl Send for PipelineLayoutId {}
unsafe impl Sync for PipelineLayoutId {}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderPipelineId(web_sys::GpuRenderPipeline);
unsafe impl Send for RenderPipelineId {}
unsafe impl Sync for RenderPipelineId {}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputePipelineId(web_sys::GpuComputePipeline);
unsafe impl Send for ComputePipelineId {}
unsafe impl Sync for ComputePipelineId {}

#[derive(Debug, Clone, PartialEq)]
pub struct CommandBufferId(web_sys::GpuCommandBuffer);
unsafe impl Send for CommandBufferId {}
unsafe impl Sync for CommandBufferId {}

#[derive(Debug, Clone, PartialEq)]
pub struct CommandEncoderId(web_sys::GpuCommandEncoder);
unsafe impl Send for CommandEncoderId {}
unsafe impl Sync for CommandEncoderId {}

#[derive(Debug, Clone, PartialEq)]
pub struct RenderPassId(web_sys::GpuRenderPassEncoder);
unsafe impl Send for RenderPassId {}
unsafe impl Sync for RenderPassId {}

#[derive(Debug, Clone, PartialEq)]
pub struct ComputePassId(web_sys::GpuComputePassEncoder);
unsafe impl Send for ComputePassId {}
unsafe impl Sync for ComputePassId {}

#[derive(Debug, Clone, PartialEq)]
pub struct QueueId(web_sys::GpuQueue);
unsafe impl Send for QueueId {}
unsafe impl Sync for QueueId {}

pub fn surface_create<W: raw_window_handle::HasRawWindowHandle>(_window: &W) -> SurfaceId {
    unimplemented! {}
}

pub fn surface_create_with_canvas(canvas: &web_sys::HtmlCanvasElement) -> SurfaceId {
    let surface_id = canvas
        .get_context("gpupresent")
        .expect("Cannot get 'gpupresent' rendering context")
        .expect("No 'gpupresent' context returned")
        .dyn_into::<web_sys::GpuCanvasContext>()
        .expect("Returned rendering context is not of type GpuCanvasContext");
    SurfaceId(surface_id)
}

pub fn enumerate_adapters(_backends: BackendBit) -> impl Iterator<Item = AdapterId> {
    std::iter::empty()
}

pub async fn request_adapter(
    options: &crate::RequestAdapterOptions<'_>,
    _backends: BackendBit,
) -> Option<AdapterId> {
    let gpu = web_sys::window()
        .map(|win| win.navigator().gpu())
        .expect("Cannot get Gpu");

    let adapter_promise = match options.power_preference {
        PowerPreference::Default => gpu.request_adapter(),
        power_preference => {
            let mut options = web_sys::GpuRequestAdapterOptions::new();
            match power_preference {
                PowerPreference::LowPower => {
                    options.power_preference(web_sys::GpuPowerPreference::LowPower);
                }
                PowerPreference::HighPerformance => {
                    options.power_preference(web_sys::GpuPowerPreference::HighPerformance);
                }
                _ => unreachable!(),
            };

            gpu.request_adapter_with_options(&options)
        }
    };

    let gpu_adapter = JsFutureSend(JsFuture::from(adapter_promise))
        .await
        .and_then(|js| js.dyn_into::<web_sys::GpuAdapter>())
        .expect("Cannot get Adapter");

    Some(AdapterId(gpu_adapter))
}

#[derive(Debug)]
pub struct DeviceDetail {
    _onuncapturederror_handler:
        wasm_bindgen::closure::Closure<dyn FnMut(web_sys::GpuUncapturedErrorEvent)>,
}
unsafe impl Send for DeviceDetail {}
unsafe impl Sync for DeviceDetail {}

pub async fn adapter_request_device(
    adapter: &AdapterId,
    desc: &DeviceDescriptor,
) -> (crate::Device, QueueId) {
    use wasm_bindgen::closure::Closure;

    let mut gpu_device_descriptor = web_sys::GpuDeviceDescriptor::new();
    gpu_device_descriptor
        .extensions(&to_gpu_extension_names(&desc.extensions))
        .limits(&to_gpu_limits(&desc.limits));

    let gpu_device: web_sys::GpuDevice = JsFutureSend(JsFuture::from(
        adapter
            .0
            .request_device_with_descriptor(&gpu_device_descriptor),
    ))
    .await
    .expect("Could not get GpuDevice")
    .dyn_into()
    .expect("value is not a GpuDevice");

    // set uncaptured error handler
    let uncaptured_error_cb = Closure::wrap(Box::new(move |e: web_sys::GpuUncapturedErrorEvent| {
        let error = e.error();
        let msg = if let Some(validation_error) = error.dyn_ref::<web_sys::GpuValidationError>() {
            format!("Validation error: {}", validation_error.message())
        } else if let Some(_) = error.dyn_ref::<web_sys::GpuOutOfMemoryError>() {
            format!("GPU out of memory")
        } else {
            format!("Unknown error type: {:?}", error)
        };
        log::warn!("Uncaptured error: {}", msg);
    })
        as Box<dyn FnMut(web_sys::GpuUncapturedErrorEvent)>);
    gpu_device.set_onuncapturederror(Some(uncaptured_error_cb.as_ref().as_ref().unchecked_ref()));

    // listen for gpu device being lost
    let gpu_device_inner = gpu_device.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let device_lost_info: web_sys::GpuDeviceLostInfo = JsFuture::from(gpu_device_inner.lost())
            .await
            .unwrap()
            .dyn_into()
            .unwrap();
        log::error!("Device lost: {}", device_lost_info.message());
    });

    let gpu_queue = gpu_device.default_queue();
    (
        crate::Device {
            id: DeviceId(gpu_device),
            temp: Default::default(),
            detail: DeviceDetail {
                _onuncapturederror_handler: uncaptured_error_cb,
            },
        },
        QueueId(gpu_queue),
    )
}

pub type BufferDetail = ();

pub struct CreateBufferMappedDetail {
    buffer_detail: BufferDetail,
    array_buffer_view: js_sys::Uint8Array,
}
pub fn create_buffer_mapped_finish(
    create_buffer_mapped: crate::CreateBufferMapped<'_>,
) -> crate::Buffer {
    unsafe {
        create_buffer_mapped
            .detail
            .array_buffer_view
            .set(&js_sys::Uint8Array::view(create_buffer_mapped.data), 0);
    }
    create_buffer_mapped.id.0.unmap();

    crate::Buffer {
        id: create_buffer_mapped.id.clone(),
        detail: create_buffer_mapped.detail.buffer_detail.clone(),
    }
}

pub fn create_buffer_mapped_drop(create_buffer_mapped: &mut crate::CreateBufferMapped<'_>) {
    unsafe {
        let _: Vec<u8> = Box::<[u8]>::from_raw(create_buffer_mapped.data).into();
    }
}

pub fn device_poll(_device: &DeviceId, _maintain: crate::Maintain) {
    // browser handles all resource cleanup and async callbacks
}

pub fn device_create_shader_module(device: &DeviceId, spv: &[u32]) -> ShaderModuleId {
    // Real hacky way to construct a web_sys::GpuShaderModuleDescriptor with a Uint32Array
    let desc_object: JsValue = js_sys::Object::from_entries(
        &js_sys::Map::new().set(&JsValue::from_str("code"), &js_sys::Uint32Array::from(spv)),
    )
    .unwrap()
    .into();
    ShaderModuleId(
        device
            .0
            .create_shader_module(&web_sys::GpuShaderModuleDescriptor::from(desc_object)),
    )
}

pub fn device_create_command_encoder(
    device: &DeviceId,
    desc: &crate::CommandEncoderDescriptor,
) -> CommandEncoderId {
    let gpu_command_encoder_descriptor = to_gpu_command_encoder_descriptor(desc);
    CommandEncoderId(
        device
            .0
            .create_command_encoder_with_descriptor(&gpu_command_encoder_descriptor),
    )
}

pub fn device_create_bind_group(
    device: &DeviceId,
    desc: &crate::BindGroupDescriptor,
) -> BindGroupId {
    BindGroupId(
        device
            .0
            .create_bind_group(&to_gpu_bind_group_descriptor(desc)),
    )
}

pub fn device_create_bind_group_layout(
    device: &DeviceId,
    desc: &crate::BindGroupLayoutDescriptor,
) -> BindGroupLayoutId {
    BindGroupLayoutId(
        device
            .0
            .create_bind_group_layout(&to_gpu_bind_group_layout_descriptor(desc)),
    )
}

pub fn device_create_pipeline_layout(
    device: &DeviceId,
    desc: &crate::PipelineLayoutDescriptor,
) -> PipelineLayoutId {
    PipelineLayoutId(
        device
            .0
            .create_pipeline_layout(&to_gpu_pipeline_layout_descriptor(desc)),
    )
}

pub fn device_create_render_pipeline(
    device: &DeviceId,
    desc: &crate::RenderPipelineDescriptor,
) -> RenderPipelineId {
    RenderPipelineId(
        device
            .0
            .create_render_pipeline(&to_gpu_render_pipeline_descriptor(desc)),
    )
}

pub fn device_create_compute_pipeline(
    device: &DeviceId,
    desc: &crate::ComputePipelineDescriptor,
) -> ComputePipelineId {
    ComputePipelineId(
        device
            .0
            .create_compute_pipeline(&to_gpu_compute_pipeline_descriptor(desc)),
    )
}

pub fn device_create_buffer(device: &DeviceId, desc: &crate::BufferDescriptor) -> crate::Buffer {
    crate::Buffer {
        id: BufferId(device.0.create_buffer(&to_gpu_buffer_descriptor(desc))),
        detail: (),
    }
}

pub fn device_create_buffer_mapped<'a, 'b>(
    device: &'a DeviceId,
    desc: &'a crate::BufferDescriptor,
) -> crate::CreateBufferMapped<'b> {
    unsafe {
        let (gpu_buffer, array_buffer_view) = {
            let gpu_mapped_buffer = device
                .0
                .create_buffer_mapped(&to_gpu_buffer_descriptor(desc));
            let array_buffer: js_sys::ArrayBuffer = gpu_mapped_buffer.get(1).dyn_into().unwrap();
            (
                BufferId(gpu_mapped_buffer.get(0).dyn_into().unwrap()),
                js_sys::Uint8Array::new(&array_buffer),
            )
        };

        let memory: Box<[u8]> = vec![0; desc.size as usize].into_boxed_slice();
        let slice: &mut [u8] =
            std::slice::from_raw_parts_mut(Box::into_raw(memory) as *mut u8, desc.size as usize);
        crate::CreateBufferMapped {
            id: gpu_buffer,
            data: slice,
            detail: CreateBufferMappedDetail {
                buffer_detail: (),
                array_buffer_view,
            },
        }
    }
}

pub fn device_create_texture(device: &DeviceId, desc: &crate::TextureDescriptor) -> TextureId {
    TextureId(device.0.create_texture(&to_gpu_texture_descriptor(desc)))
}

pub fn device_create_sampler(device: &DeviceId, desc: &crate::SamplerDescriptor) -> SamplerId {
    SamplerId(
        device
            .0
            .create_sampler_with_descriptor(&to_gpu_sampler_descriptor(desc)),
    )
}

pub fn device_create_swap_chain(
    device: &DeviceId,
    surface: &SurfaceId,
    desc: &SwapChainDescriptor,
) -> SwapChainId {
    SwapChainId(
        surface
            .0
            .configure_swap_chain(&to_gpu_swap_chain_descriptor(device, desc)),
    )
}

pub fn device_drop(_device: &DeviceId) {}

pub fn bind_group_drop(_bind_group: &BindGroupId) {}

pub struct BufferReadMappingDetail {
    data: Vec<u8>,
}

pub fn buffer_read_mapping_as_slice<'a>(
    buffer_read_mapping: &'a crate::BufferReadMapping,
) -> &'a [u8] {
    &buffer_read_mapping.detail.data
}

pub fn buffer_map_read(
    buffer: &crate::Buffer,
    start: BufferAddress,
    size: BufferAddress,
) -> impl Future<Output = Result<crate::BufferReadMapping, crate::BufferAsyncErr>> + Send {
    // web does not support mapping with an offset
    assert_eq!(start, 0);

    let buffer_id = buffer.id.clone();
    let map_future = JsFutureSend(JsFuture::from(buffer.id.0.map_read_async()));

    async move {
        let array_buffer = map_future.await.map_err(|_| crate::BufferAsyncErr)?;
        let mut data = vec![0; size as usize];
        js_sys::Uint8Array::new(&array_buffer).copy_to(&mut data);

        Ok(crate::BufferReadMapping {
            buffer_id: buffer_id,
            detail: { BufferReadMappingDetail { data } },
        })
    }
}

pub struct BufferWriteMappingDetail {
    data: Vec<u8>,
    array_buffer_view: js_sys::Uint8Array,
}

pub fn buffer_write_mapping_as_slice<'a>(
    buffer_write_mapping: &'a mut crate::BufferWriteMapping,
) -> &'a mut [u8] {
    &mut buffer_write_mapping.detail.data
}

pub fn buffer_write_mapping_unmap(buffer_write_mapping: &mut crate::BufferWriteMapping) {
    unsafe {
        buffer_write_mapping.detail.array_buffer_view.set(
            &js_sys::Uint8Array::view(&buffer_write_mapping.detail.data),
            0,
        )
    }
    buffer_write_mapping.buffer_id.0.unmap();
}

/// Map the buffer for writing. The result is returned in a future.
pub fn buffer_map_write(
    buffer: &crate::Buffer,
    start: BufferAddress,
    size: BufferAddress,
) -> impl Future<Output = Result<crate::BufferWriteMapping, crate::BufferAsyncErr>> + Send {
    // web does not support mapping with an offset
    assert_eq!(start, 0);

    let buffer_id = buffer.id.clone();
    let map_future = JsFutureSend(JsFuture::from(buffer.id.0.map_write_async()));

    async move {
        let array_buffer = map_future.await.map_err(|_| crate::BufferAsyncErr)?;
        let array_buffer_view = js_sys::Uint8Array::new(&array_buffer);
        let data = vec![0; size as usize];

        Ok(crate::BufferWriteMapping {
            buffer_id: buffer_id,
            detail: BufferWriteMappingDetail {
                data,
                array_buffer_view,
            },
        })
    }
}

pub fn buffer_unmap(buffer: &BufferId) {
    buffer.0.unmap()
}

pub fn buffer_destroy(_buffer: &BufferId) {
    // Let javascript GC clean up buffers
}

pub fn texture_create_view(
    texture: &TextureId,
    desc: &crate::TextureViewDescriptor,
) -> TextureViewId {
    TextureViewId(
        texture
            .0
            .create_view_with_descriptor(&to_gpu_texture_view_descriptor(desc)),
    )
}

pub fn texture_create_default_view(texture: &TextureId) -> TextureViewId {
    TextureViewId(texture.0.create_view())
}

pub fn texture_destroy(_texture: &TextureId) {
    // Let javascript GC clean up textures
}

pub fn texture_view_destroy(_texture_view: &TextureViewId) {
    // presumably automatically destroyed on web
}

pub fn command_encoder_finish(command_encoder: CommandEncoderId) -> CommandBufferId {
    CommandBufferId(command_encoder.0.finish())
}

pub fn command_encoder_begin_render_pass(
    command_encoder: &CommandEncoderId,
    desc: &crate::RenderPassDescriptor<'_, '_>,
) -> RenderPassId {
    RenderPassId(
        command_encoder
            .0
            .begin_render_pass(&to_gpu_render_pass_descriptor(desc)),
    )
}

pub fn command_encoder_begin_compute_pass(command_encoder: &CommandEncoderId) -> ComputePassId {
    ComputePassId(command_encoder.0.begin_compute_pass())
}

pub fn command_encoder_copy_buffer_to_buffer(
    command_encoder: &CommandEncoderId,
    source: &BufferId,
    source_offset: BufferAddress,
    destination: &BufferId,
    destination_offset: BufferAddress,
    copy_size: BufferAddress,
) {
    command_encoder
        .0
        .copy_buffer_to_buffer_with_u32_and_u32_and_u32(
            &source.0,
            source_offset as u32,
            &destination.0,
            destination_offset as u32,
            copy_size as u32,
        );
}

pub fn command_encoder_copy_buffer_to_texture(
    command_encoder: &CommandEncoderId,
    source: crate::BufferCopyView,
    destination: crate::TextureCopyView,
    copy_size: crate::Extent3d,
) {
    let gpu_source = to_gpu_buffer_copy_view(&source);
    let gpu_destination = to_gpu_texture_copy_view(&destination);
    let gpu_copy_size = to_gpu_extent_3d_dict(copy_size);
    command_encoder
        .0
        .copy_buffer_to_texture_with_gpu_extent_3d_dict(
            &gpu_source,
            &gpu_destination,
            &gpu_copy_size,
        );
}

pub fn command_encoder_copy_texture_to_buffer(
    command_encoder: &CommandEncoderId,
    source: crate::TextureCopyView,
    destination: crate::BufferCopyView,
    copy_size: crate::Extent3d,
) {
    command_encoder
        .0
        .copy_texture_to_buffer_with_gpu_extent_3d_dict(
            &to_gpu_texture_copy_view(&source),
            &to_gpu_buffer_copy_view(&destination),
            &to_gpu_extent_3d_dict(copy_size),
        );
}

pub fn command_encoder_copy_texture_to_texture(
    command_encoder: &CommandEncoderId,
    source: crate::TextureCopyView,
    destination: crate::TextureCopyView,
    copy_size: crate::Extent3d,
) {
    command_encoder
        .0
        .copy_texture_to_texture_with_gpu_extent_3d_dict(
            &to_gpu_texture_copy_view(&source),
            &to_gpu_texture_copy_view(&destination),
            &to_gpu_extent_3d_dict(copy_size),
        );
}

pub fn render_pass_set_bind_group(
    render_pass: &RenderPassId,
    index: u32,
    bind_group: &BindGroupId,
    offsets: &[DynamicOffset],
) {
    render_pass.0.set_bind_group_with_u32_sequence(
        index,
        &bind_group.0,
        &offsets
            .iter()
            .map(|v| JsValue::from_f64(*v as f64))
            .collect::<js_sys::Array>(),
    )
}

pub fn render_pass_set_pipeline(render_pass: &RenderPassId, pipeline: &RenderPipelineId) {
    render_pass.0.set_pipeline(&pipeline.0)
}

pub fn render_pass_set_blend_color(render_pass: &RenderPassId, color: Color) {
    render_pass
        .0
        .set_blend_color_with_f64_sequence(&to_gpu_color_array(&color));
}

pub fn render_pass_set_index_buffer(
    render_pass: &RenderPassId,
    buffer: &BufferId,
    offset: BufferAddress,
    _size: BufferAddress,
) {
    render_pass
        .0
        .set_index_buffer_with_u32(&buffer.0, offset as u32)
}

pub fn render_pass_set_vertex_buffer(
    render_pass: &RenderPassId,
    slot: u32,
    buffer: &BufferId,
    offset: BufferAddress,
    _size: BufferAddress,
) {
    render_pass
        .0
        .set_vertex_buffer_with_u32(slot, &buffer.0, offset as u32);
}

pub fn render_pass_set_scissor_rect(render_pass: &RenderPassId, x: u32, y: u32, w: u32, h: u32) {
    render_pass.0.set_scissor_rect(x, y, w, h)
}

pub fn render_pass_set_viewport(
    render_pass: &RenderPassId,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    min_depth: f32,
    max_depth: f32,
) {
    render_pass.0.set_viewport(x, y, w, h, min_depth, max_depth)
}

pub fn render_pass_set_stencil_reference(render_pass: &RenderPassId, reference: u32) {
    render_pass.0.set_stencil_reference(reference)
}

pub fn render_pass_draw(render_pass: &RenderPassId, vertices: Range<u32>, instances: Range<u32>) {
    render_pass.0.draw(
        vertices.end - vertices.start,
        instances.end - instances.start,
        vertices.start,
        instances.start,
    )
}

pub fn render_pass_draw_indexed(
    render_pass: &RenderPassId,
    indices: Range<u32>,
    base_vertex: i32,
    instances: Range<u32>,
) {
    render_pass.0.draw_indexed(
        indices.end - indices.start,
        instances.end - instances.start,
        indices.start,
        base_vertex,
        instances.start,
    )
}

pub fn render_pass_draw_indirect(
    render_pass: &RenderPassId,
    indirect_buffer: &BufferId,
    indirect_offset: BufferAddress,
) {
    render_pass
        .0
        .draw_indirect_with_u32(&indirect_buffer.0, indirect_offset as u32);
}

pub fn render_pass_draw_indexed_indirect(
    render_pass: &RenderPassId,
    indirect_buffer: &BufferId,
    indirect_offset: BufferAddress,
) {
    render_pass
        .0
        .draw_indexed_indirect_with_u32(&indirect_buffer.0, indirect_offset as u32)
}

pub fn render_pass_end_pass(render_pass: &RenderPassId) {
    render_pass.0.end_pass()
}

/// Sets the active bind group for a given bind group index.
pub fn compute_pass_set_bind_group(
    compute_pass: &ComputePassId,
    index: u32,
    bind_group: &BindGroupId,
    offsets: &[DynamicOffset],
) {
    compute_pass.0.set_bind_group_with_u32_sequence(
        index,
        &bind_group.0,
        &offsets
            .iter()
            .map(|offset| JsValue::from_f64(*offset as f64))
            .collect::<js_sys::Array>(),
    )
}

pub fn compute_pass_set_pipeline(compute_pass: &ComputePassId, pipeline: &ComputePipelineId) {
    compute_pass.0.set_pipeline(&pipeline.0)
}

pub fn compute_pass_dispatch(compute_pass: &ComputePassId, x: u32, y: u32, z: u32) {
    compute_pass.0.dispatch_with_y_and_z(x, y, z)
}

pub fn compute_pass_dispatch_indirect(
    compute_pass: &ComputePassId,
    indirect_buffer: &BufferId,
    indirect_offset: BufferAddress,
) {
    compute_pass
        .0
        .dispatch_indirect_with_u32(&indirect_buffer.0, indirect_offset as u32)
}

pub fn compute_pass_end_pass(compute_pass: &ComputePassId) {
    compute_pass.0.end_pass()
}

pub fn queue_submit(queue: &QueueId, command_buffers: &[crate::CommandBuffer]) {
    let buffers_array = command_buffers
        .iter()
        .map(|buffer| (&buffer.id.0 as &JsValue).clone())
        .collect::<js_sys::Array>();
    queue.0.submit(&buffers_array);
}

pub fn swap_chain_present(_swap_chain: &SwapChainId) {
    // presents automatically on web
}

pub fn swap_chain_get_next_texture(
    swap_chain: &SwapChainId,
) -> Result<TextureViewId, crate::TimeOut> {
    Ok(TextureViewId(
        swap_chain.0.get_current_texture().create_view(),
    ))
}

// mapping to web types

fn to_gpu_extension_names(extensions: &Extensions) -> JsValue {
    let Extensions {
        // no web_sys::GpuExtensionName enum variant matches this extension
        anisotropic_filtering: _,
    } = extensions;

    // no Extensions variant matches for the following web_sys::GpuExtensionName variants:
    // - GpuExtensionName::TextureCompressionBc

    // No variants in Extensions currently map to any variants in GpuExtensionName, just create an
    // empty array
    js_sys::Array::new().dyn_into().unwrap()
}

fn to_gpu_limits(limits: &Limits) -> web_sys::GpuLimits {
    let Limits { max_bind_groups } = limits;

    let mut gpu_limits = web_sys::GpuLimits::new();
    gpu_limits.max_bind_groups(*max_bind_groups);
    gpu_limits
}

fn to_gpu_command_encoder_descriptor(
    desc: &crate::CommandEncoderDescriptor,
) -> web_sys::GpuCommandEncoderDescriptor {
    let crate::CommandEncoderDescriptor { label } = desc;

    // there are no fields to map here
    let mut gpu_command_encoder = web_sys::GpuCommandEncoderDescriptor::new();
    if let Some(label) = label {
        gpu_command_encoder.label(label);
    }
    gpu_command_encoder
}

fn to_gpu_render_pass_descriptor(
    desc: &crate::RenderPassDescriptor,
) -> web_sys::GpuRenderPassDescriptor {
    use crate::RenderPassDescriptor;
    let RenderPassDescriptor {
        color_attachments,
        depth_stencil_attachment,
    } = desc;

    let gpu_color_attachments = js_sys::Array::new_with_length(color_attachments.len() as u32);
    for (i, color_attachment_desc) in color_attachments.iter().enumerate() {
        gpu_color_attachments.set(
            i as u32,
            to_gpu_render_pass_color_attachment_descriptor(color_attachment_desc)
                .dyn_into()
                .unwrap(),
        );
    }

    let mut gpu_render_pass_descriptor =
        web_sys::GpuRenderPassDescriptor::new(&gpu_color_attachments);
    if let Some(depth_stencil_attachment_desc) = depth_stencil_attachment.as_ref() {
        gpu_render_pass_descriptor.depth_stencil_attachment(
            &to_gpu_depth_stencil_attachment_descriptor(depth_stencil_attachment_desc),
        );
    }

    gpu_render_pass_descriptor
}

fn to_gpu_render_pass_color_attachment_descriptor(
    desc: &crate::RenderPassColorAttachmentDescriptor,
) -> web_sys::GpuRenderPassColorAttachmentDescriptor {
    let mut gpu_render_pass_color_attachment_descriptor =
        web_sys::GpuRenderPassColorAttachmentDescriptor::new(
            &desc.attachment.id.0,
            &to_gpu_load_color_value(desc.load_op, &desc.clear_color),
        );
    gpu_render_pass_color_attachment_descriptor.store_op(to_gpu_store_op(desc.store_op));
    if let Some(resolve_target) = desc.resolve_target {
        gpu_render_pass_color_attachment_descriptor.resolve_target(&resolve_target.id.0);
    }
    gpu_render_pass_color_attachment_descriptor
}

fn to_gpu_depth_stencil_attachment_descriptor(
    desc: &crate::RenderPassDepthStencilAttachmentDescriptor,
) -> web_sys::GpuRenderPassDepthStencilAttachmentDescriptor {
    web_sys::GpuRenderPassDepthStencilAttachmentDescriptor::new(
        &desc.attachment.id.0,
        &to_gpu_load_depth_value(desc.depth_load_op, desc.clear_depth),
        to_gpu_store_op(desc.depth_store_op),
        &to_gpu_load_stencil_value(desc.stencil_load_op, desc.clear_stencil),
        to_gpu_store_op(desc.stencil_store_op),
    )
}

fn to_gpu_load_color_value(load_op: LoadOp, clear_color: &Color) -> JsValue {
    match load_op {
        // Clearing with a GpuColorDict seems to not work, use an array instead
        // LoadOp::Clear => to_gpu_color_dict(clear_color)
        //     .dyn_into()
        //     .expect("Could not convert GpuColorDict into JsValue"),
        LoadOp::Clear => to_gpu_color_array(clear_color).dyn_into().unwrap(),
        LoadOp::Load => JsValue::from_str("load"),
    }
}

fn to_gpu_load_depth_value(load_op: LoadOp, clear_depth: f32) -> JsValue {
    match load_op {
        LoadOp::Clear => JsValue::from_f64(clear_depth as f64),
        LoadOp::Load => JsValue::from_str("load"),
    }
}

fn to_gpu_load_stencil_value(load_op: LoadOp, clear_stencil: u32) -> JsValue {
    match load_op {
        LoadOp::Clear => JsValue::from_f64(clear_stencil as f64),
        LoadOp::Load => JsValue::from_str("load"),
    }
}

fn to_gpu_store_op(store_op: StoreOp) -> web_sys::GpuStoreOp {
    match store_op {
        StoreOp::Clear => web_sys::GpuStoreOp::Clear,
        StoreOp::Store => web_sys::GpuStoreOp::Store,
    }
}

#[allow(unused)]
fn to_gpu_color_dict(color: &Color) -> web_sys::GpuColorDict {
    web_sys::GpuColorDict::new(color.r, color.g, color.b, color.a)
}

fn to_gpu_color_array(color: &Color) -> js_sys::Array {
    [color.r, color.g, color.b, color.a]
        .iter()
        .map(|v| JsValue::from_f64(*v))
        .collect::<js_sys::Array>()
}

fn to_gpu_texture_component_type(
    component_type: TextureComponentType,
) -> web_sys::GpuTextureComponentType {
    match component_type {
        TextureComponentType::Float => web_sys::GpuTextureComponentType::Float,
        TextureComponentType::Sint => web_sys::GpuTextureComponentType::Sint,
        TextureComponentType::Uint => web_sys::GpuTextureComponentType::Uint,
    }
}

fn to_gpu_texture_format(format: TextureFormat) -> web_sys::GpuTextureFormat {
    match format {
        // Normal 8 bit formats
        TextureFormat::R8Unorm => web_sys::GpuTextureFormat::R8unorm,
        TextureFormat::R8Snorm => web_sys::GpuTextureFormat::R8snorm,
        TextureFormat::R8Uint => web_sys::GpuTextureFormat::R8uint,
        TextureFormat::R8Sint => web_sys::GpuTextureFormat::R8sint,
        TextureFormat::R16Uint => web_sys::GpuTextureFormat::R16uint,
        TextureFormat::R16Sint => web_sys::GpuTextureFormat::R16sint,
        TextureFormat::R16Float => web_sys::GpuTextureFormat::R16float,
        TextureFormat::Rg8Unorm => web_sys::GpuTextureFormat::Rg8unorm,
        TextureFormat::Rg8Snorm => web_sys::GpuTextureFormat::Rg8snorm,
        TextureFormat::Rg8Uint => web_sys::GpuTextureFormat::Rg8uint,
        TextureFormat::Rg8Sint => web_sys::GpuTextureFormat::Rg8sint,
        TextureFormat::R32Uint => web_sys::GpuTextureFormat::R32uint,
        TextureFormat::R32Sint => web_sys::GpuTextureFormat::R32sint,
        TextureFormat::R32Float => web_sys::GpuTextureFormat::R32float,
        TextureFormat::Rg16Uint => web_sys::GpuTextureFormat::Rg16uint,
        TextureFormat::Rg16Sint => web_sys::GpuTextureFormat::Rg16sint,
        TextureFormat::Rg16Float => web_sys::GpuTextureFormat::Rg16float,
        TextureFormat::Rgba8Unorm => web_sys::GpuTextureFormat::Rgba8unorm,
        TextureFormat::Rgba8UnormSrgb => web_sys::GpuTextureFormat::Rgba8unormSrgb,
        TextureFormat::Rgba8Snorm => web_sys::GpuTextureFormat::Rgba8snorm,
        TextureFormat::Rgba8Uint => web_sys::GpuTextureFormat::Rgba8uint,
        TextureFormat::Rgba8Sint => web_sys::GpuTextureFormat::Rgba8sint,
        TextureFormat::Bgra8Unorm => web_sys::GpuTextureFormat::Bgra8unorm,
        TextureFormat::Bgra8UnormSrgb => web_sys::GpuTextureFormat::Bgra8unormSrgb,
        TextureFormat::Rgb10a2Unorm => web_sys::GpuTextureFormat::Rgb10a2unorm,
        TextureFormat::Rg11b10Float => web_sys::GpuTextureFormat::Rg11b10float,
        TextureFormat::Rg32Uint => web_sys::GpuTextureFormat::Rg32uint,
        TextureFormat::Rg32Sint => web_sys::GpuTextureFormat::Rg32sint,
        TextureFormat::Rg32Float => web_sys::GpuTextureFormat::Rg32float,
        TextureFormat::Rgba16Uint => web_sys::GpuTextureFormat::Rgba16uint,
        TextureFormat::Rgba16Sint => web_sys::GpuTextureFormat::Rgba16sint,
        TextureFormat::Rgba16Float => web_sys::GpuTextureFormat::Rgba16float,
        TextureFormat::Rgba32Uint => web_sys::GpuTextureFormat::Rgba32uint,
        TextureFormat::Rgba32Sint => web_sys::GpuTextureFormat::Rgba32sint,
        TextureFormat::Rgba32Float => web_sys::GpuTextureFormat::Rgba32float,
        TextureFormat::Depth32Float => web_sys::GpuTextureFormat::Depth32float,
        TextureFormat::Depth24Plus => web_sys::GpuTextureFormat::Depth24plus,
        TextureFormat::Depth24PlusStencil8 => web_sys::GpuTextureFormat::Depth24plusStencil8,
    }
}

fn to_gpu_swap_chain_descriptor(
    device: &DeviceId,
    desc: &SwapChainDescriptor,
) -> web_sys::GpuSwapChainDescriptor {
    let SwapChainDescriptor {
        usage,
        format,
        width: _,
        height: _,
        present_mode: _,
    } = desc;

    let mut gpu_swap_chain_descriptor =
        web_sys::GpuSwapChainDescriptor::new(&device.0, to_gpu_texture_format(*format));
    gpu_swap_chain_descriptor.usage(usage.bits());
    gpu_swap_chain_descriptor
}

fn to_gpu_bind_group_entry(binding: &crate::Binding) -> web_sys::GpuBindGroupEntry {
    use crate::BindingResource;
    let resource: JsValue = match binding.resource {
        BindingResource::Buffer { buffer, ref range } => {
            let mut gpu_buffer_binding = web_sys::GpuBufferBinding::new(&buffer.id.0);
            gpu_buffer_binding.offset(range.start as f64);
            gpu_buffer_binding.size((range.end - range.start) as f64);
            gpu_buffer_binding.into()
        }
        BindingResource::Sampler(sampler) => sampler.id.clone().0.into(),
        BindingResource::TextureView(texture_view) => texture_view.id.0.clone().into(),
    };
    web_sys::GpuBindGroupEntry::new(binding.binding, &resource)
}

fn to_gpu_buffer_descriptor(desc: &crate::BufferDescriptor) -> web_sys::GpuBufferDescriptor {
    let crate::BufferDescriptor { size, usage, label } = desc;
    let mut gpu_buffer_descriptor = web_sys::GpuBufferDescriptor::new(*size as f64, usage.bits());
    if let Some(label) = label {
        gpu_buffer_descriptor.label(label);
    }
    gpu_buffer_descriptor
}

fn to_gpu_bind_group_descriptor(
    desc: &crate::BindGroupDescriptor,
) -> web_sys::GpuBindGroupDescriptor {
    use crate::BindGroupDescriptor;
    let BindGroupDescriptor {
        layout,
        bindings,
        label,
    } = desc;
    let gpu_bindings = bindings
        .iter()
        .map(to_gpu_bind_group_entry)
        .collect::<js_sys::Array>();
    let mut gpu_bind_group_descriptor =
        web_sys::GpuBindGroupDescriptor::new(&gpu_bindings, &layout.id.0);
    if let Some(label) = label {
        gpu_bind_group_descriptor.label(label);
    }
    gpu_bind_group_descriptor
}

fn to_gpu_index_format(index_format: IndexFormat) -> web_sys::GpuIndexFormat {
    match index_format {
        IndexFormat::Uint16 => web_sys::GpuIndexFormat::Uint16,
        IndexFormat::Uint32 => web_sys::GpuIndexFormat::Uint32,
    }
}

fn to_gpu_vertex_state_descriptor(
    index_format: IndexFormat,
    vertex_buffers: &[crate::VertexBufferDescriptor],
) -> web_sys::GpuVertexStateDescriptor {
    let mut gpu_vertex_state_descriptor = web_sys::GpuVertexStateDescriptor::new();

    gpu_vertex_state_descriptor.index_format(to_gpu_index_format(index_format));
    let gpu_vertex_buffers = vertex_buffers
        .iter()
        .map(to_gpu_vertex_buffer_layout_descriptor)
        .collect::<js_sys::Array>();
    gpu_vertex_state_descriptor.vertex_buffers(&gpu_vertex_buffers);

    gpu_vertex_state_descriptor
}

fn to_gpu_input_step_mode(mode: InputStepMode) -> web_sys::GpuInputStepMode {
    match mode {
        InputStepMode::Vertex => web_sys::GpuInputStepMode::Vertex,
        InputStepMode::Instance => web_sys::GpuInputStepMode::Instance,
    }
}

fn to_gpu_vertex_buffer_layout_descriptor(
    desc: &crate::VertexBufferDescriptor,
) -> web_sys::GpuVertexBufferLayoutDescriptor {
    use crate::VertexBufferDescriptor;
    let VertexBufferDescriptor {
        stride,
        step_mode,
        attributes,
    } = desc;

    let gpu_vertex_buffer_attributes = attributes
        .iter()
        .map(to_gpu_vertex_attribute_descriptor)
        .collect::<js_sys::Array>();
    let mut gpu_vertex_buffer_layout_descriptor = web_sys::GpuVertexBufferLayoutDescriptor::new(
        *stride as f64,
        &gpu_vertex_buffer_attributes,
    );
    gpu_vertex_buffer_layout_descriptor.step_mode(to_gpu_input_step_mode(*step_mode));
    gpu_vertex_buffer_layout_descriptor
}

fn to_gpu_compute_pipeline_descriptor(
    desc: &crate::ComputePipelineDescriptor,
) -> web_sys::GpuComputePipelineDescriptor {
    use crate::ComputePipelineDescriptor;
    let ComputePipelineDescriptor {
        layout,
        compute_stage,
    } = desc;

    web_sys::GpuComputePipelineDescriptor::new(
        &layout.id.0,
        &to_gpu_programmable_stage_descriptor(compute_stage),
    )
}

fn to_gpu_render_pipeline_descriptor(
    desc: &crate::RenderPipelineDescriptor,
) -> web_sys::GpuRenderPipelineDescriptor {
    use crate::RenderPipelineDescriptor;
    let RenderPipelineDescriptor {
        layout,
        vertex_stage,
        fragment_stage,
        rasterization_state,
        primitive_topology,
        color_states,
        depth_stencil_state,
        vertex_state:
            crate::VertexStateDescriptor {
                index_format,
                vertex_buffers,
            },
        sample_count,
        sample_mask,
        alpha_to_coverage_enabled,
    } = desc;

    let gpu_color_states = color_states
        .iter()
        .map(to_gpu_color_state_descriptor)
        .collect::<js_sys::Array>();
    let mut gpu_render_pipeline_descriptor = web_sys::GpuRenderPipelineDescriptor::new(
        &layout.id.0,
        &gpu_color_states,
        to_gpu_primitive_topology(*primitive_topology),
        &to_gpu_programmable_stage_descriptor(vertex_stage),
    );

    if let Some(fragment_stage) = fragment_stage {
        gpu_render_pipeline_descriptor
            .fragment_stage(&to_gpu_programmable_stage_descriptor(fragment_stage));
    }
    if let Some(rasterization_state) = rasterization_state {
        gpu_render_pipeline_descriptor
            .rasterization_state(&to_gpu_rasterization_state_descriptor(rasterization_state));
    }
    if let Some(depth_stencil_state) = depth_stencil_state {
        gpu_render_pipeline_descriptor
            .depth_stencil_state(&to_gpu_depth_stencil_state_descriptor(depth_stencil_state));
    }

    gpu_render_pipeline_descriptor.vertex_state(&to_gpu_vertex_state_descriptor(
        *index_format,
        vertex_buffers,
    ));

    gpu_render_pipeline_descriptor.sample_count(*sample_count);
    gpu_render_pipeline_descriptor.sample_mask(*sample_mask);
    gpu_render_pipeline_descriptor.alpha_to_coverage_enabled(*alpha_to_coverage_enabled);

    gpu_render_pipeline_descriptor
}

fn to_gpu_vertex_format(vertex_format: VertexFormat) -> web_sys::GpuVertexFormat {
    match vertex_format {
        VertexFormat::Uchar2 => web_sys::GpuVertexFormat::Uchar2,
        VertexFormat::Uchar4 => web_sys::GpuVertexFormat::Uchar4,
        VertexFormat::Char2 => web_sys::GpuVertexFormat::Char2,
        VertexFormat::Char4 => web_sys::GpuVertexFormat::Char4,
        VertexFormat::Uchar2Norm => web_sys::GpuVertexFormat::Uchar2norm,
        VertexFormat::Uchar4Norm => web_sys::GpuVertexFormat::Uchar4norm,
        VertexFormat::Char2Norm => web_sys::GpuVertexFormat::Char2norm,
        VertexFormat::Char4Norm => web_sys::GpuVertexFormat::Char4norm,
        VertexFormat::Ushort2 => web_sys::GpuVertexFormat::Ushort2,
        VertexFormat::Ushort4 => web_sys::GpuVertexFormat::Ushort4,
        VertexFormat::Short2 => web_sys::GpuVertexFormat::Short2,
        VertexFormat::Short4 => web_sys::GpuVertexFormat::Short4,
        VertexFormat::Ushort2Norm => web_sys::GpuVertexFormat::Ushort2norm,
        VertexFormat::Ushort4Norm => web_sys::GpuVertexFormat::Ushort4norm,
        VertexFormat::Short2Norm => web_sys::GpuVertexFormat::Short2norm,
        VertexFormat::Short4Norm => web_sys::GpuVertexFormat::Short4norm,
        VertexFormat::Half2 => web_sys::GpuVertexFormat::Half2,
        VertexFormat::Half4 => web_sys::GpuVertexFormat::Half4,
        VertexFormat::Float => web_sys::GpuVertexFormat::Float,
        VertexFormat::Float2 => web_sys::GpuVertexFormat::Float2,
        VertexFormat::Float3 => web_sys::GpuVertexFormat::Float3,
        VertexFormat::Float4 => web_sys::GpuVertexFormat::Float4,
        VertexFormat::Uint => web_sys::GpuVertexFormat::Uint,
        VertexFormat::Uint2 => web_sys::GpuVertexFormat::Uint2,
        VertexFormat::Uint3 => web_sys::GpuVertexFormat::Uint3,
        VertexFormat::Uint4 => web_sys::GpuVertexFormat::Uint4,
        VertexFormat::Int => web_sys::GpuVertexFormat::Int,
        VertexFormat::Int2 => web_sys::GpuVertexFormat::Int2,
        VertexFormat::Int3 => web_sys::GpuVertexFormat::Int3,
        VertexFormat::Int4 => web_sys::GpuVertexFormat::Int4,
    }
}

fn to_gpu_vertex_attribute_descriptor(
    desc: &VertexAttributeDescriptor,
) -> web_sys::GpuVertexAttributeDescriptor {
    let VertexAttributeDescriptor {
        offset,
        format,
        shader_location,
    } = *desc;

    web_sys::GpuVertexAttributeDescriptor::new(
        to_gpu_vertex_format(format),
        offset as f64,
        shader_location,
    )
}

fn to_gpu_compare_function(compare: CompareFunction) -> Option<web_sys::GpuCompareFunction> {
    match compare {
        CompareFunction::Never => Some(web_sys::GpuCompareFunction::Never),
        CompareFunction::Less => Some(web_sys::GpuCompareFunction::Less),
        CompareFunction::Equal => Some(web_sys::GpuCompareFunction::Equal),
        CompareFunction::LessEqual => Some(web_sys::GpuCompareFunction::LessEqual),
        CompareFunction::Greater => Some(web_sys::GpuCompareFunction::Greater),
        CompareFunction::NotEqual => Some(web_sys::GpuCompareFunction::NotEqual),
        CompareFunction::GreaterEqual => Some(web_sys::GpuCompareFunction::GreaterEqual),
        CompareFunction::Always => Some(web_sys::GpuCompareFunction::Always),
        CompareFunction::Undefined => None,
    }
}

fn to_gpu_stencil_operation(stencil_operation: &StencilOperation) -> web_sys::GpuStencilOperation {
    match stencil_operation {
        StencilOperation::Keep => web_sys::GpuStencilOperation::Keep,
        StencilOperation::Zero => web_sys::GpuStencilOperation::Zero,
        StencilOperation::Replace => web_sys::GpuStencilOperation::Replace,
        StencilOperation::Invert => web_sys::GpuStencilOperation::Invert,
        StencilOperation::IncrementClamp => web_sys::GpuStencilOperation::IncrementClamp,
        StencilOperation::DecrementClamp => web_sys::GpuStencilOperation::DecrementClamp,
        StencilOperation::IncrementWrap => web_sys::GpuStencilOperation::IncrementWrap,
        StencilOperation::DecrementWrap => web_sys::GpuStencilOperation::DecrementWrap,
    }
}

fn to_gpu_stencil_state_face_descriptor(
    desc: &StencilStateFaceDescriptor,
) -> web_sys::GpuStencilStateFaceDescriptor {
    let StencilStateFaceDescriptor {
        compare,
        fail_op,
        depth_fail_op,
        pass_op,
    } = desc;

    let mut gpu_stencil_state_face_descriptor = web_sys::GpuStencilStateFaceDescriptor::new();

    if let Some(compare) = to_gpu_compare_function(*compare) {
        gpu_stencil_state_face_descriptor.compare(compare);
    }
    gpu_stencil_state_face_descriptor.fail_op(to_gpu_stencil_operation(fail_op));
    gpu_stencil_state_face_descriptor.depth_fail_op(to_gpu_stencil_operation(depth_fail_op));
    gpu_stencil_state_face_descriptor.pass_op(to_gpu_stencil_operation(pass_op));

    gpu_stencil_state_face_descriptor
}

fn to_gpu_depth_stencil_state_descriptor(
    desc: &DepthStencilStateDescriptor,
) -> web_sys::GpuDepthStencilStateDescriptor {
    let DepthStencilStateDescriptor {
        format,
        depth_write_enabled,
        depth_compare,
        stencil_front,
        stencil_back,
        stencil_read_mask,
        stencil_write_mask,
    } = desc;

    let mut gpu_depth_stencil_state_descriptor =
        web_sys::GpuDepthStencilStateDescriptor::new(to_gpu_texture_format(*format));

    gpu_depth_stencil_state_descriptor.depth_write_enabled(*depth_write_enabled);
    if let Some(depth_compare) = to_gpu_compare_function(*depth_compare) {
        gpu_depth_stencil_state_descriptor.depth_compare(depth_compare);
    }
    gpu_depth_stencil_state_descriptor
        .stencil_front(&to_gpu_stencil_state_face_descriptor(stencil_front));
    gpu_depth_stencil_state_descriptor
        .stencil_back(&to_gpu_stencil_state_face_descriptor(stencil_back));
    gpu_depth_stencil_state_descriptor.stencil_read_mask(*stencil_read_mask);
    gpu_depth_stencil_state_descriptor.stencil_write_mask(*stencil_write_mask);

    gpu_depth_stencil_state_descriptor
}

fn to_gpu_color_state_descriptor(desc: &ColorStateDescriptor) -> web_sys::GpuColorStateDescriptor {
    let ColorStateDescriptor {
        format,
        alpha_blend,
        color_blend,
        write_mask,
    } = desc;

    let mut gpu_color_state_descriptor =
        web_sys::GpuColorStateDescriptor::new(to_gpu_texture_format(*format));

    gpu_color_state_descriptor.alpha_blend(&to_gpu_blend_descriptor(alpha_blend));
    gpu_color_state_descriptor.color_blend(&to_gpu_blend_descriptor(color_blend));
    gpu_color_state_descriptor.write_mask(write_mask.bits());

    gpu_color_state_descriptor
}

fn to_gpu_blend_factor(blend_factor: BlendFactor) -> web_sys::GpuBlendFactor {
    match blend_factor {
        BlendFactor::Zero => web_sys::GpuBlendFactor::Zero,
        BlendFactor::One => web_sys::GpuBlendFactor::One,
        BlendFactor::SrcColor => web_sys::GpuBlendFactor::SrcColor,
        BlendFactor::OneMinusSrcColor => web_sys::GpuBlendFactor::OneMinusSrcColor,
        BlendFactor::SrcAlpha => web_sys::GpuBlendFactor::SrcAlpha,
        BlendFactor::OneMinusSrcAlpha => web_sys::GpuBlendFactor::OneMinusSrcAlpha,
        BlendFactor::DstColor => web_sys::GpuBlendFactor::DstColor,
        BlendFactor::OneMinusDstColor => web_sys::GpuBlendFactor::OneMinusDstColor,
        BlendFactor::DstAlpha => web_sys::GpuBlendFactor::DstAlpha,
        BlendFactor::OneMinusDstAlpha => web_sys::GpuBlendFactor::OneMinusDstAlpha,
        BlendFactor::SrcAlphaSaturated => web_sys::GpuBlendFactor::SrcAlphaSaturated,
        BlendFactor::BlendColor => web_sys::GpuBlendFactor::BlendColor,
        BlendFactor::OneMinusBlendColor => web_sys::GpuBlendFactor::OneMinusBlendColor,
    }
}

fn to_gpu_blend_operation(blend_operation: BlendOperation) -> web_sys::GpuBlendOperation {
    match blend_operation {
        BlendOperation::Add => web_sys::GpuBlendOperation::Add,
        BlendOperation::Subtract => web_sys::GpuBlendOperation::Subtract,
        BlendOperation::ReverseSubtract => web_sys::GpuBlendOperation::ReverseSubtract,
        BlendOperation::Min => web_sys::GpuBlendOperation::Min,
        BlendOperation::Max => web_sys::GpuBlendOperation::Max,
    }
}

fn to_gpu_blend_descriptor(desc: &BlendDescriptor) -> web_sys::GpuBlendDescriptor {
    let BlendDescriptor {
        src_factor,
        dst_factor,
        operation,
    } = *desc;

    let mut gpu_blend_descriptor = web_sys::GpuBlendDescriptor::new();

    gpu_blend_descriptor.src_factor(to_gpu_blend_factor(src_factor));
    gpu_blend_descriptor.dst_factor(to_gpu_blend_factor(dst_factor));
    gpu_blend_descriptor.operation(to_gpu_blend_operation(operation));

    gpu_blend_descriptor
}

fn to_gpu_programmable_stage_descriptor(
    desc: &crate::ProgrammableStageDescriptor,
) -> web_sys::GpuProgrammableStageDescriptor {
    web_sys::GpuProgrammableStageDescriptor::new(&desc.entry_point, &desc.module.id.0)
}

fn to_gpu_cull_mode(cull_mode: CullMode) -> web_sys::GpuCullMode {
    match cull_mode {
        CullMode::None => web_sys::GpuCullMode::None,
        CullMode::Front => web_sys::GpuCullMode::Front,
        CullMode::Back => web_sys::GpuCullMode::Back,
    }
}

fn to_gpu_front_face(front_face: FrontFace) -> web_sys::GpuFrontFace {
    match front_face {
        FrontFace::Ccw => web_sys::GpuFrontFace::Ccw,
        FrontFace::Cw => web_sys::GpuFrontFace::Cw,
    }
}

fn to_gpu_rasterization_state_descriptor(
    desc: &RasterizationStateDescriptor,
) -> web_sys::GpuRasterizationStateDescriptor {
    let RasterizationStateDescriptor {
        front_face,
        cull_mode,
        depth_bias,
        depth_bias_slope_scale,
        depth_bias_clamp,
    } = *desc;

    let mut gpu_rasterization_state_descriptor = web_sys::GpuRasterizationStateDescriptor::new();

    gpu_rasterization_state_descriptor.cull_mode(to_gpu_cull_mode(cull_mode));
    gpu_rasterization_state_descriptor.depth_bias(depth_bias);
    gpu_rasterization_state_descriptor.depth_bias_clamp(depth_bias_clamp);
    gpu_rasterization_state_descriptor.depth_bias_slope_scale(depth_bias_slope_scale);
    gpu_rasterization_state_descriptor.front_face(to_gpu_front_face(front_face));

    gpu_rasterization_state_descriptor
}

fn to_gpu_primitive_topology(
    primitive_topology: PrimitiveTopology,
) -> web_sys::GpuPrimitiveTopology {
    match primitive_topology {
        PrimitiveTopology::PointList => web_sys::GpuPrimitiveTopology::PointList,
        PrimitiveTopology::LineList => web_sys::GpuPrimitiveTopology::LineList,
        PrimitiveTopology::LineStrip => web_sys::GpuPrimitiveTopology::LineStrip,
        PrimitiveTopology::TriangleList => web_sys::GpuPrimitiveTopology::TriangleList,
        PrimitiveTopology::TriangleStrip => web_sys::GpuPrimitiveTopology::TriangleStrip,
    }
}

fn to_gpu_texture_view_dimension(
    dimension: TextureViewDimension,
) -> web_sys::GpuTextureViewDimension {
    match dimension {
        TextureViewDimension::D1 => web_sys::GpuTextureViewDimension::N1d,
        TextureViewDimension::D2 => web_sys::GpuTextureViewDimension::N2d,
        TextureViewDimension::D2Array => web_sys::GpuTextureViewDimension::N2dArray,
        TextureViewDimension::Cube => web_sys::GpuTextureViewDimension::Cube,
        TextureViewDimension::CubeArray => web_sys::GpuTextureViewDimension::CubeArray,
        TextureViewDimension::D3 => web_sys::GpuTextureViewDimension::N3d,
    }
}

fn to_gpu_shader_stage(shader_stage: ShaderStage) -> u32 {
    shader_stage.bits()
}

fn to_gpu_bind_group_layout_binding(
    binding: &crate::BindGroupLayoutEntry,
) -> web_sys::GpuBindGroupLayoutEntry {
    use crate::{BindGroupLayoutEntry, BindingType};
    let BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
    } = binding;

    let visibility = to_gpu_shader_stage(*visibility);
    match ty {
        BindingType::UniformBuffer { dynamic } => {
            let mut gpu_bind_group_layout_binding = web_sys::GpuBindGroupLayoutEntry::new(
                *binding,
                web_sys::GpuBindingType::UniformBuffer,
                visibility,
            );
            if *dynamic {
                gpu_bind_group_layout_binding.has_dynamic_offset(true);
            }
            gpu_bind_group_layout_binding
        }
        BindingType::StorageBuffer { readonly, dynamic } => {
            let mut gpu_bind_group_layout_binding = if *readonly {
                web_sys::GpuBindGroupLayoutEntry::new(
                    *binding,
                    web_sys::GpuBindingType::ReadonlyStorageBuffer,
                    visibility,
                )
            } else {
                web_sys::GpuBindGroupLayoutEntry::new(
                    *binding,
                    web_sys::GpuBindingType::StorageBuffer,
                    visibility,
                )
            };
            if *dynamic {
                gpu_bind_group_layout_binding.has_dynamic_offset(true);
            }
            gpu_bind_group_layout_binding
        }
        BindingType::Sampler { comparison } => web_sys::GpuBindGroupLayoutEntry::new(
            *binding,
            if *comparison {
                web_sys::GpuBindingType::ComparisonSampler
            } else {
                web_sys::GpuBindingType::Sampler
            },
            visibility,
        ),
        BindingType::SampledTexture {
            multisampled,
            component_type,
            dimension,
        } => {
            let mut gpu_bind_group_layout_binding = web_sys::GpuBindGroupLayoutEntry::new(
                *binding,
                web_sys::GpuBindingType::SampledTexture,
                visibility,
            );
            if *multisampled {
                gpu_bind_group_layout_binding.multisampled(true);
            }
            gpu_bind_group_layout_binding.view_dimension(to_gpu_texture_view_dimension(*dimension));
            gpu_bind_group_layout_binding
                .texture_component_type(to_gpu_texture_component_type(*component_type));
            gpu_bind_group_layout_binding
        }
        BindingType::StorageTexture {
            dimension,
            format,
            component_type,
            readonly,
        } => {
            let mut gpu_bind_group_layout_binding = web_sys::GpuBindGroupLayoutEntry::new(
                *binding,
                if *readonly {
                    web_sys::GpuBindingType::ReadonlyStorageTexture
                } else {
                    web_sys::GpuBindingType::WriteonlyStorageTexture
                },
                visibility,
            );
            gpu_bind_group_layout_binding.view_dimension(to_gpu_texture_view_dimension(*dimension));
            gpu_bind_group_layout_binding.storage_texture_format(to_gpu_texture_format(*format));
            gpu_bind_group_layout_binding
                .texture_component_type(to_gpu_texture_component_type(*component_type));
            gpu_bind_group_layout_binding
        }
    }
}

fn to_gpu_bind_group_layout_descriptor(
    desc: &crate::BindGroupLayoutDescriptor,
) -> web_sys::GpuBindGroupLayoutDescriptor {
    let crate::BindGroupLayoutDescriptor { bindings, label } = desc;
    let mut gpu_bind_group_layout_descriptor = web_sys::GpuBindGroupLayoutDescriptor::new(
        &bindings
            .iter()
            .map(to_gpu_bind_group_layout_binding)
            .collect::<js_sys::Array>(),
    );
    if let Some(label) = label {
        gpu_bind_group_layout_descriptor.label(label);
    }
    gpu_bind_group_layout_descriptor
}

fn to_gpu_pipeline_layout_descriptor(
    desc: &crate::PipelineLayoutDescriptor,
) -> web_sys::GpuPipelineLayoutDescriptor {
    web_sys::GpuPipelineLayoutDescriptor::new(
        &desc
            .bind_group_layouts
            .iter()
            .map(|layout| &layout.id.0)
            .collect::<js_sys::Array>(),
    )
}

fn to_gpu_texture_view_descriptor(
    desc: &crate::TextureViewDescriptor,
) -> web_sys::GpuTextureViewDescriptor {
    use crate::TextureViewDescriptor;
    let TextureViewDescriptor {
        format,
        dimension,
        aspect,
        base_mip_level,
        level_count,
        base_array_layer,
        array_layer_count,
    } = desc;

    let mut gpu_texture_view_descriptor = web_sys::GpuTextureViewDescriptor::new();

    gpu_texture_view_descriptor.format(to_gpu_texture_format(*format));
    gpu_texture_view_descriptor.dimension(to_gpu_texture_view_dimension(*dimension));
    gpu_texture_view_descriptor.aspect(to_gpu_texture_aspect(*aspect));
    gpu_texture_view_descriptor.base_mip_level(*base_mip_level);
    gpu_texture_view_descriptor.mip_level_count(*level_count);
    gpu_texture_view_descriptor.base_array_layer(*base_array_layer);
    gpu_texture_view_descriptor.array_layer_count(*array_layer_count);

    gpu_texture_view_descriptor
}

fn to_gpu_texture_aspect(aspect: crate::TextureAspect) -> web_sys::GpuTextureAspect {
    use crate::TextureAspect;
    match aspect {
        TextureAspect::All => web_sys::GpuTextureAspect::All,
        TextureAspect::StencilOnly => web_sys::GpuTextureAspect::StencilOnly,
        TextureAspect::DepthOnly => web_sys::GpuTextureAspect::DepthOnly,
    }
}

fn to_gpu_sampler_descriptor(desc: &SamplerDescriptor) -> web_sys::GpuSamplerDescriptor {
    let SamplerDescriptor {
        address_mode_u,
        address_mode_v,
        address_mode_w,
        mag_filter,
        min_filter,
        mipmap_filter,
        lod_min_clamp,
        lod_max_clamp,
        compare,
    } = desc;

    let mut gpu_sampler_descriptor = web_sys::GpuSamplerDescriptor::new();
    gpu_sampler_descriptor.address_mode_u(to_gpu_address_mode(*address_mode_u));
    gpu_sampler_descriptor.address_mode_v(to_gpu_address_mode(*address_mode_v));
    gpu_sampler_descriptor.address_mode_w(to_gpu_address_mode(*address_mode_w));
    gpu_sampler_descriptor.mag_filter(to_gpu_filter_mode(*mag_filter));
    gpu_sampler_descriptor.min_filter(to_gpu_filter_mode(*min_filter));
    gpu_sampler_descriptor.mipmap_filter(to_gpu_filter_mode(*mipmap_filter));
    gpu_sampler_descriptor.lod_min_clamp(*lod_min_clamp);
    gpu_sampler_descriptor.lod_max_clamp(*lod_max_clamp);
    if let Some(compare) = to_gpu_compare_function(*compare) {
        gpu_sampler_descriptor.compare(compare);
    }

    gpu_sampler_descriptor
}

fn to_gpu_address_mode(mode: crate::AddressMode) -> web_sys::GpuAddressMode {
    use crate::AddressMode;
    match mode {
        AddressMode::ClampToEdge => web_sys::GpuAddressMode::ClampToEdge,
        AddressMode::Repeat => web_sys::GpuAddressMode::Repeat,
        AddressMode::MirrorRepeat => web_sys::GpuAddressMode::MirrorRepeat,
    }
}

fn to_gpu_filter_mode(mode: crate::FilterMode) -> web_sys::GpuFilterMode {
    use crate::FilterMode;
    match mode {
        FilterMode::Nearest => web_sys::GpuFilterMode::Nearest,
        FilterMode::Linear => web_sys::GpuFilterMode::Linear,
    }
}

fn to_gpu_texture_copy_view(view: &crate::TextureCopyView) -> web_sys::GpuTextureCopyView {
    use crate::TextureCopyView;
    let TextureCopyView {
        texture,
        mip_level,
        array_layer,
        origin,
    } = view;

    let mut gpu_texture_copy_view = web_sys::GpuTextureCopyView::new(&texture.id.0);
    gpu_texture_copy_view.array_layer(*array_layer);
    gpu_texture_copy_view.mip_level(*mip_level);
    gpu_texture_copy_view.origin(&to_gpu_origin_3d_dict(origin));
    gpu_texture_copy_view
}

fn to_gpu_origin_3d_dict(origin: &crate::Origin3d) -> web_sys::GpuOrigin3dDict {
    let mut gpu_origin_3d_dict = web_sys::GpuOrigin3dDict::new();
    gpu_origin_3d_dict.x(origin.x as u32);
    gpu_origin_3d_dict.y(origin.y as u32);
    gpu_origin_3d_dict.z(origin.z as u32);
    gpu_origin_3d_dict
}

fn to_gpu_buffer_copy_view(view: &crate::BufferCopyView) -> web_sys::GpuBufferCopyView {
    use crate::BufferCopyView;
    let BufferCopyView {
        buffer,
        offset,
        bytes_per_row,
        rows_per_image,
    } = view;

    let mut gpu_buffer_view = web_sys::GpuBufferCopyView::new(&buffer.id.0, *bytes_per_row);
    gpu_buffer_view.image_height(*rows_per_image);
    gpu_buffer_view.offset(*offset as f64);
    gpu_buffer_view
}

fn to_gpu_texture_descriptor(desc: &crate::TextureDescriptor) -> web_sys::GpuTextureDescriptor {
    use crate::TextureDescriptor;
    let TextureDescriptor {
        label,
        size,
        array_layer_count,
        mip_level_count,
        sample_count,
        dimension,
        format,
        usage,
    } = desc;

    let mut gpu_texture_descriptor = web_sys::GpuTextureDescriptor::new(
        to_gpu_texture_format(*format),
        &to_gpu_extent_3d_dict(*size),
        usage.bits(),
    );

    gpu_texture_descriptor.array_layer_count(*array_layer_count);
    gpu_texture_descriptor.mip_level_count(*mip_level_count);
    gpu_texture_descriptor.sample_count(*sample_count);
    gpu_texture_descriptor.dimension(to_gpu_texture_dimension(*dimension));
    if let Some(label) = label {
        gpu_texture_descriptor.label(label);
    }

    gpu_texture_descriptor
}

fn to_gpu_extent_3d_dict(extent: crate::Extent3d) -> web_sys::GpuExtent3dDict {
    web_sys::GpuExtent3dDict::new(extent.depth, extent.height, extent.width)
}

fn to_gpu_texture_dimension(
    texture_dimension: crate::TextureDimension,
) -> web_sys::GpuTextureDimension {
    use crate::TextureDimension;
    match texture_dimension {
        TextureDimension::D1 => web_sys::GpuTextureDimension::N1d,
        TextureDimension::D2 => web_sys::GpuTextureDimension::N2d,
        TextureDimension::D3 => web_sys::GpuTextureDimension::N3d,
    }
}

// Wrapper for JsFuture to make it Send
pub struct JsFutureSend(JsFuture);

// This is safe only because this module is only built for the wasm32 target which doesn't support
// threads. If built for a target which does support threads this is NOT SOUND.
unsafe impl Send for JsFutureSend {}

impl Future for JsFutureSend {
    type Output = Result<JsValue, JsValue>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        unsafe { Future::poll(self.map_unchecked_mut(|f| &mut f.0), cx) }
    }
}
