/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/plugin/aluminum_shark/platform.h"

#include <iostream>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/plugin/aluminum_shark/executor.h"
#include "tensorflow/compiler/plugin/aluminum_shark/logging.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {
namespace aluminum_shark {

XlaAluminumSharkPlatform::XlaAluminumSharkPlatform(const std::string& name,
                                                   const Platform::Id& id)
    : name_(name), id_(id) {}

XlaAluminumSharkPlatform::~XlaAluminumSharkPlatform() {}

Platform::Id XlaAluminumSharkPlatform::id() const { return id_; }

int XlaAluminumSharkPlatform::VisibleDeviceCount() const { return 1; }

const std::string& XlaAluminumSharkPlatform::Name() const { return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
XlaAluminumSharkPlatform::DescriptionForDevice(int ordinal) const {
  return XlaAluminumSharkExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> XlaAluminumSharkPlatform::ExecutorForDevice(
    int ordinal) {
  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*>
XlaAluminumSharkPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> XlaAluminumSharkPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
XlaAluminumSharkPlatform::GetUncachedExecutor(
    const StreamExecutorConfig& config) {
  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<XlaAluminumSharkExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

void XlaAluminumSharkPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
  LOG(FATAL) << "not yet implemented: register executor trace listener";
}

void XlaAluminumSharkPlatform::UnregisterTraceListener(
    TraceListener* listener) {
  LOG(FATAL) << "not yet implemented: unregister executor trace listener";
}

static void InitializeXlaAluminumSharkPlatform() {
  AS_LOG_S << "running platform init" << std::endl;
  std::unique_ptr<Platform> platform(new XlaAluminumSharkPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));

  std::vector<stream_executor::Platform*> platforms =
      xla::PlatformUtil::GetSupportedPlatforms().ValueOrDie();
  for (size_t i = 0; i < platforms.size(); ++i) {
    AS_LOG_S << "platform: " << platforms[i]->Name() << std::endl;
    std::unique_ptr<DeviceDescription> device_desc =
        platforms[i]->DescriptionForDevice(-1).ValueOrDie();
    AS_LOG_S << "device: " << device_desc->name() << std::endl;
  }
}

}  // namespace aluminum_shark
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    aluminum_shark_platform,
    stream_executor::aluminum_shark::InitializeXlaAluminumSharkPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(aluminum_shark_platform,
                                     multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     aluminum_shark_platform);
