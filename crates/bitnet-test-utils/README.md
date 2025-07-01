# BitNet Test Utils (`bitnet-test-utils`)

This crate provides robust test utilities for the BitNet workspace, including the `TestReporter` utility for detailed, thread-safe test reporting and markdown log generation. It is used by core, converter, and kernel tests to ensure comprehensive validation and reporting.

- **TestReporter**: Thread-safe, in-memory and markdown test reporting utility.
- **Usage**: Used in all major test suites for logging, timing, and generating detailed reports.

No public API is guaranteed stable; this crate is for internal workspace use. 