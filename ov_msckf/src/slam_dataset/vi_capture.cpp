#include "vi_capture.h"

#include <thread>

namespace slam_dataset {

void ViCapture::waitStreamingOver(double polling_period_ms) {
  while (isStreaming()) {
    std::this_thread::sleep_for(
        std::chrono::milliseconds(polling_period_ms));
  }
}
}  // namespace slam_dataset

