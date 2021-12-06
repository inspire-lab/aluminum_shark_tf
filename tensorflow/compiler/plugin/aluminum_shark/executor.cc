#include "tensorflow/compiler/plugin/aluminum_shark/executor.h"

#include <cstring>

#include "tensorflow/compiler/xla/status_macros.h"

namespace stream_executor {
namespace aluminum_shark {

host::HostStream* AsExecutorStream(Stream* stream) {
  DCHECK(stream != nullptr);
  return dynamic_cast<host::HostStream*>(stream->implementation());
}

XlaAluminumSharkExecutor::XlaAluminumSharkExecutor(
    const PluginConfig& plugin_config)
    : plugin_config_(plugin_config) {}

XlaAluminumSharkExecutor::~XlaAluminumSharkExecutor() {}

DeviceMemoryBase XlaAluminumSharkExecutor::Allocate(uint64 size,
                                                    int64_t memory_space) {
  return DeviceMemoryBase(new char[size], size);
}

void* XlaAluminumSharkExecutor::GetSubBuffer(DeviceMemoryBase* parent,
                                             uint64 offset_bytes,
                                             uint64 /*size_bytes*/) {
  return parent + offset_bytes;
}

void XlaAluminumSharkExecutor::Deallocate(DeviceMemoryBase* mem) {
  delete[] static_cast<char*>(mem->opaque());
}

bool XlaAluminumSharkExecutor::Memcpy(Stream* stream, void* host_dst,
                                      const DeviceMemoryBase& dev_src,
                                      uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, host_dst, dev_src, size]() {
    port::Status ok = SynchronousMemcpy(host_dst, dev_src, size);
  });
  AsExecutorStream(stream)->BlockUntilDone();
  return true;
}

bool XlaAluminumSharkExecutor::Memcpy(Stream* stream, DeviceMemoryBase* dev_dst,
                                      const void* host_src, uint64 size) {
  AsExecutorStream(stream)->EnqueueTask([this, dev_dst, host_src, size]() {
    port::Status ok = SynchronousMemcpy(dev_dst, host_src, size);
  });
  AsExecutorStream(stream)->BlockUntilDone();
  return true;
}

port::Status XlaAluminumSharkExecutor::SynchronousMemcpy(
    DeviceMemoryBase* dev_dst, const void* host_src, uint64 size) {
  memcpy(dev_dst->opaque(), host_src, size);
  return port::Status::OK();
}

port::Status XlaAluminumSharkExecutor::SynchronousMemcpy(
    void* host_dst, const DeviceMemoryBase& dev_src, uint64 size) {
  memcpy(host_dst, dev_src.opaque(), size);
  return port::Status::OK();
}

bool XlaAluminumSharkExecutor::HostCallback(
    Stream* stream, std::function<port::Status()> callback) {
  AsExecutorStream(stream)->EnqueueTask([callback]() {
    port::Status s = callback();
    if (!s.ok()) {
      LOG(WARNING) << "Host callback failed: " << s;
    }
  });
  return true;
}

bool XlaAluminumSharkExecutor::CreateStreamDependency(Stream* dependent,
                                                      Stream* other) {
  AsExecutorStream(dependent)->EnqueueTask(
      [other]() { SE_CHECK_OK(other->BlockHostUntilDone()); });
  AsExecutorStream(dependent)->BlockUntilDone();
  return true;
}

bool XlaAluminumSharkExecutor::StartTimer(Stream* stream, Timer* timer) {
  dynamic_cast<host::HostTimer*>(timer->implementation())->Start(stream);
  return true;
}

bool XlaAluminumSharkExecutor::StopTimer(Stream* stream, Timer* timer) {
  dynamic_cast<host::HostTimer*>(timer->implementation())->Stop(stream);
  return true;
}

port::Status XlaAluminumSharkExecutor::BlockHostUntilDone(Stream* stream) {
  AsExecutorStream(stream)->BlockUntilDone();
  return port::Status::OK();
}

port::StatusOr<std::unique_ptr<DeviceDescription>>
XlaAluminumSharkExecutor::CreateDeviceDescription(int device_ordinal) {
  internal::DeviceDescriptionBuilder builder;

  builder.set_device_address_bits(64);

  builder.set_name("AluminumShark");
  builder.set_device_memory_size(static_cast<uint64_t>(4) * 1024 * 1024 * 1024);
  builder.set_clock_rate_ghz(static_cast<float>(CLOCKS_PER_SEC) / 1e9);

  return builder.Build();
}

}  // namespace aluminum_shark
}  // namespace stream_executor
