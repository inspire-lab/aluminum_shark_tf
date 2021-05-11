/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace data {
namespace {

class ExecutorTest : public ::testing::Test {
 protected:
  ExecutorTest()
      : device_(DeviceFactory::NewDevice("CPU", {},
                                         "/job:localhost/replica:0/task:0")) {}

  ~ExecutorTest() override {
    // There should always be exactly one Ref left on the Rendezvous
    // when the test completes.
    CHECK(rendez_->Unref());
  }

  // Resets executor_ with a new executor based on a graph 'gdef'.
  void Create(std::unique_ptr<const Graph> graph) {
    const int version = graph->versions().producer();
    LocalExecutorParams params;
    params.device = device_.get();
    params.create_kernel =
        [this, version](const std::shared_ptr<const NodeProperties>& props,
                        OpKernel** kernel) {
          return CreateNonCachedKernel(device_.get(), nullptr, props, version,
                                       kernel);
        };
    params.delete_kernel = [](OpKernel* kernel) {
      DeleteNonCachedKernel(kernel);
    };
    TF_CHECK_OK(
        NewExecutor("SINGLE_THREADED_EXECUTOR", params, *graph, &exec_));
    runner_ = [](const std::function<void()>& fn) { fn(); };
    rendez_ = NewLocalRendezvous();
  }

  Status Run(Rendezvous* rendez) {
    Executor::Args args;
    args.rendezvous = rendez;
    args.runner = runner_;
    return exec_->Run(args);
  }

  Status Run(CallFrameInterface* call_frame) {
    Executor::Args args;
    args.call_frame = call_frame;
    args.runner = runner_;
    return exec_->Run(args);
  }

  std::unique_ptr<Device> device_;
  std::unique_ptr<Executor> exec_ = nullptr;
  Executor::Args::Runner runner_;
  Rendezvous* rendez_ = nullptr;
};

// A float val -> Tensor<float>
Tensor V(const float val) {
  Tensor tensor(DT_FLOAT, TensorShape({}));
  tensor.scalar<float>()() = val;
  return tensor;
}

// Tensor<float> -> a float val.
float V(const Tensor& tensor) {
  CHECK_EQ(tensor.dtype(), DT_FLOAT);
  CHECK(TensorShapeUtils::IsScalar(tensor.shape()));
  return tensor.scalar<float>()();
}

Rendezvous::ParsedKey Key(const string& sender, const uint64 incarnation,
                          const string& receiver, const string& name) {
  Rendezvous::ParsedKey result;
  TF_CHECK_OK(
      Rendezvous::ParseKey(Rendezvous::CreateKey(sender, incarnation, receiver,
                                                 name, FrameAndIter(0, 0)),
                           &result));
  return result;
}

TEST_F(ExecutorTest, SimpleAdd) {
  // c = a + b
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Arg(g.get(), 0, DT_FLOAT);
  auto in1 = test::graph::Arg(g.get(), 1, DT_FLOAT);
  auto tmp = test::graph::Add(g.get(), in0, in1);
  auto ret = test::graph::Retval(g.get(), 0, tmp);
  g->AddControlEdge(in1, ret);
  FixupSourceAndSinkEdges(g.get());
  Create(std::move(g));
  FunctionCallFrame call_frame({DT_FLOAT, DT_FLOAT}, {DT_FLOAT});
  TF_ASSERT_OK(call_frame.SetArgs({V(1.0), V(2.0)}));
  TF_ASSERT_OK(Run(&call_frame));
  std::vector<Tensor> retvals;
  TF_ASSERT_OK(call_frame.ConsumeRetvals(&retvals, false));
  EXPECT_EQ(3.0, V(retvals[0]));  // out = 1.0 + 2.0 = 3.0

  // Verify that the argument values are unchanged.
  const Tensor* arg_0;
  TF_ASSERT_OK(call_frame.GetArg(0, &arg_0));
  EXPECT_EQ(1.0, V(*arg_0));
  const Tensor* arg_1;
  TF_ASSERT_OK(call_frame.GetArg(1, &arg_1));
  EXPECT_EQ(2.0, V(*arg_1));
}

TEST_F(ExecutorTest, SelfAdd) {
  // v0 <- a
  // v1 = v0 + v0
  // v2 = v1 + v1
  // ... ...
  // v10 = v9 + v9
  //
  // b <- v10
  // All nodes are executed by one thread.
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto v = test::graph::Arg(g.get(), 0, DT_FLOAT);
  const int N = 10;
  for (int i = 1; i <= N; ++i) {
    v = test::graph::Add(g.get(), v, v);
  }
  // out <- v10
  test::graph::Retval(g.get(), 0, v);
  FixupSourceAndSinkEdges(g.get());
  Create(std::move(g));
  FunctionCallFrame call_frame({DT_FLOAT}, {DT_FLOAT});
  // a = 1.0
  TF_ASSERT_OK(call_frame.SetArgs({V(1.0)}));
  TF_ASSERT_OK(Run(&call_frame));
  std::vector<Tensor> retvals;
  TF_ASSERT_OK(call_frame.ConsumeRetvals(&retvals, false));
  EXPECT_EQ(1024.0, V(retvals[0]));  // b=v10=2*v9=4*v8=...=1024*a=1024.0
}

// Builds a graph which adds N copies of one variable "in". I.e.,
//     a + a + a + ... + a
// The returned graph is parenthesized ramdonly. I.e.,
//     a + ((a + a) + a)
//     (a + a) + (a + a)
//     ((a + a) + a) + a
// are all possibly generated.
void BuildTree(int N, Graph* g) {
  CHECK_GT(N, 1);
  // A single input node "in".
  auto in = test::graph::Arg(g, 0, DT_FLOAT);
  std::vector<Node*> nodes;
  int i = 0;
  // Duplicate "in" N times. Each copies is named as l0, l1, l2, ....
  for (; i < N; ++i) {
    nodes.push_back(test::graph::Identity(g, in, 0));
  }
  random::PhiloxRandom philox(0, 17);
  random::SimplePhilox rnd(&philox);
  while (nodes.size() > 1) {
    // Randomly pick two from nodes and add them. The resulting node
    // is named lik n10, n11, .... and is put back into "nodes".
    int x = rnd.Uniform(nodes.size());
    auto in0 = nodes[x];
    nodes[x] = nodes.back();
    nodes.resize(nodes.size() - 1);
    x = rnd.Uniform(nodes.size());
    auto in1 = nodes[x];
    // node = in0 + in1.
    nodes[x] = test::graph::Add(g, in0, in1);
  }
  // The final output node "out".
  test::graph::Retval(g, 0, nodes.back());
  FixupSourceAndSinkEdges(g);
}

TEST_F(ExecutorTest, RandomTree) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  BuildTree(4096, g.get());
  Create(std::move(g));
  FunctionCallFrame call_frame({DT_FLOAT}, {DT_FLOAT});
  TF_ASSERT_OK(call_frame.SetArgs({V(1.0)}));
  TF_ASSERT_OK(Run(&call_frame));
  std::vector<Tensor> retvals;
  TF_ASSERT_OK(call_frame.ConsumeRetvals(&retvals, false));
  EXPECT_EQ(4096.0, V(retvals[0]));
}

TEST_F(ExecutorTest, OpError) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto zero = test::graph::Constant(g.get(), V(0.0));
  auto inf = test::graph::Unary(g.get(), "Reciprocal", zero);
  auto check = test::graph::CheckNumerics(g.get(), inf, "message");
  auto two = test::graph::Constant(g.get(), V(2.0));
  test::graph::Binary(g.get(), "Mul", check, two);
  FixupSourceAndSinkEdges(g.get());
  Create(std::move(g));
  FunctionCallFrame call_frame({}, {});
  // Fails due to invalid dtype.
  EXPECT_TRUE(errors::IsInvalidArgument(Run(&call_frame)));
}

TEST_F(ExecutorTest, ControlDependenciesFromSpecialNodes) {
  auto g = absl::make_unique<Graph>(OpRegistry::Global());
  auto in0 = test::graph::Arg(g.get(), 0, DT_FLOAT);
  auto one = test::graph::Constant(g.get(), V(2.0));
  auto add = test::graph::Add(g.get(), in0, one);
  auto ret = test::graph::Retval(g.get(), 0, add);
  g->AddControlEdge(in0, add);
  g->AddControlEdge(one, ret);
  FixupSourceAndSinkEdges(g.get());
  Create(std::move(g));
  FunctionCallFrame call_frame({DT_FLOAT}, {DT_FLOAT});
  TF_ASSERT_OK(call_frame.SetArgs({V(1.0)}));
  TF_ASSERT_OK(Run(&call_frame));
  std::vector<Tensor> retvals;
  TF_ASSERT_OK(call_frame.ConsumeRetvals(&retvals, false));
  EXPECT_EQ(3.0, V(retvals[0]));  // out = 1.0 + 2.0 = 3.0
}

void BM_executor(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int depth = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());
  random::PhiloxRandom philox(1729, 17);
  random::SimplePhilox rand(&philox);
  uint64 cur = 0;
  uint32 r = 1 + rand.Rand32() % width;
  std::vector<Node*> ready_nodes;
  for (int i = 0; i < r; ++i) {
    ready_nodes.push_back(test::graph::NoOp(g, {}));
    ++cur;
  }
  std::random_device random_device;
  std::mt19937 rng(random_device());
  for (int i = 0; i < depth; ++i) {
    std::shuffle(ready_nodes.begin(), ready_nodes.end(), rng);
    r = 1 + rand.Rand32() % (ready_nodes.size());
    std::vector<Node*> control_inputs;
    for (int j = 0; j < r; ++j) {
      control_inputs.push_back(ready_nodes.back());
      ready_nodes.pop_back();
    }
    Node* n = test::graph::NoOp(g, control_inputs);
    ++cur;
    r = 1 + rand.Rand32() % width;
    for (int j = 0; j < r; ++j) {
      ready_nodes.push_back(test::graph::NoOp(g, {n}));
      ++cur;
    }
  }
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api=*/false)
      .Run(state);
  state.SetLabel(strings::StrCat("Nodes = ", cur));
  state.SetItemsProcessed(cur * static_cast<int64>(state.iterations()));
}

// Tall skinny graphs
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(16, 1024);
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(32, 8192);

// Short fat graphs
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(1024, 16);
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(8192, 32);

// Tall fat graph
BENCHMARK(BM_executor)->UseRealTime()->ArgPair(1024, 1024);

void BM_const_identity(::testing::benchmark::State& state) {
  const int width = state.range(0);
  const int outputs_per_const = state.range(1);

  Graph* g = new Graph(OpRegistry::Global());
  for (int i = 0; i < width; ++i) {
    Tensor i_t(i);
    Node* const_node = test::graph::Constant(g, i_t);
    for (int j = 0; j < outputs_per_const; ++j) {
      test::graph::Identity(g, const_node);
    }
  }
  FixupSourceAndSinkEdges(g);
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR",
                  /*old_benchmark_api=*/false)
      .Run(state);
  state.SetLabel(strings::StrCat("Nodes = ", (1 + outputs_per_const) * width));
  state.SetItemsProcessed((1 + outputs_per_const) * width *
                          static_cast<int64>(state.iterations()));
}

// Graph with actual op execution.
BENCHMARK(BM_const_identity)->UseRealTime()->ArgPair(1, 1);
BENCHMARK(BM_const_identity)->UseRealTime()->ArgPair(1, 100);
BENCHMARK(BM_const_identity)->UseRealTime()->ArgPair(100, 1);
BENCHMARK(BM_const_identity)->UseRealTime()->ArgPair(100, 100);

// TODO(mrry): This benchmark currently crashes with a use-after free, because
// test::Benchmark::RunWithArgs() assumes that the executor will take ownership
// of the given graph, *and* keep its nodes (`x`, `y` and `z`) alive for the
// duration of the benchmark. Since the single threaded executor does not retain
// a copy of the graph, this fails.
//
// TODO(mrry): Add support for Arg/Retval "function call convention" in
// `test::Benchmark::RunWithArgs()`.
#if 0
#define ALICE "/job:j/replica:0/task:0/cpu:0"
#define BOB "/job:j/replica:0/task:0/gpu:0"

static void BM_FeedInputFetchOutput(::testing::benchmark::State& state) {
  Graph* g = new Graph(OpRegistry::Global());
  // z = x + y: x and y are provided as benchmark inputs.  z is the
  // output of the benchmark.  Conceptually, the caller is ALICE, the
  // benchmark is BOB.
  Node* x = test::graph::Recv(g, "x", "float", ALICE, 1, BOB);
  Node* y = test::graph::Recv(g, "y", "float", ALICE, 1, BOB);
  Node* sum = test::graph::Add(g, x, y);
  Node* z = test::graph::Send(g, sum, "z", BOB, 1, ALICE);
  FixupSourceAndSinkEdges(g);
  Tensor val(DT_FLOAT, TensorShape({}));
  val.scalar<float>()() = 3.14;
  test::Benchmark("cpu", g, nullptr, nullptr, nullptr,
                  "SINGLE_THREADED_EXECUTOR", /*old_benchmark_api=*/false)
      .RunWithArgs({{x, val}, {y, val}}, {z}, state);
  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FeedInputFetchOutput);
#endif

}  // namespace
}  // namespace data
}  // namespace tensorflow