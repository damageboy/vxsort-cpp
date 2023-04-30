#include <cstdio>
#include <backward.hpp>

#include "gtest/gtest.h"

#if defined(GTEST_OS_ESP8266) || defined(GTEST_OS_ESP32)
// Arduino-like platforms: program entry points are setup/loop instead of main.

#ifdef GTEST_OS_ESP8266
extern "C" {
#endif

void setup() { testing::InitGoogleTest(); }

void loop() { RUN_ALL_TESTS(); }

#ifdef GTEST_OS_ESP8266
}
#endif

#elif defined(GTEST_OS_QURT)
// QuRT: program entry point is main, but argc/argv are unusable.

GTEST_API_ int main() {
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
#else
// Normal platforms: program entry point is main, argc/argv are initialized.

GTEST_API_ int main(int argc, char **argv) {
    backward::SignalHandling sh;

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif