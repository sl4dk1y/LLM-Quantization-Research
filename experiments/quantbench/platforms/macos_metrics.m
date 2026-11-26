#import <Foundation/Foundation.h>
#import <libproc.h>
#import <dispatch/dispatch.h>
#include <sys/sysctl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc == 2 && strcmp(argv[1], "--pressure-watch") == 0) {
            dispatch_queue_t queue = dispatch_queue_create("quantbench.memory-pressure", DISPATCH_QUEUE_SERIAL);
            dispatch_source_t source = dispatch_source_create(
                DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
                DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN |
                    DISPATCH_MEMORYPRESSURE_CRITICAL, queue);
            if (source == NULL) return 3;
            dispatch_source_set_event_handler(source, ^{
                unsigned long flags = dispatch_source_get_data(source);
                const char *state = (flags & DISPATCH_MEMORYPRESSURE_CRITICAL) ? "critical" :
                                    (flags & DISPATCH_MEMORYPRESSURE_WARN) ? "warning" : "normal";
                printf("{\"event\":\"%s\"}\n", state);
                fflush(stdout);
            });
            dispatch_resume(source);
            dispatch_main();
            return 0;
        }
        if (argc != 2) {
            fprintf(stderr, "usage: macos_metrics PID\n");
            return 2;
        }
        char *end = NULL;
        long raw_pid = strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || raw_pid < 0 || raw_pid > INT32_MAX) {
            fprintf(stderr, "invalid PID\n");
            return 2;
        }
        int32_t pid = (int32_t)raw_pid;
        rusage_info_current usage = {0};
        int status = pid > 0 ? proc_pid_rusage(pid, RUSAGE_INFO_CURRENT, (rusage_info_t *)&usage) : -1;
        NSProcessInfoThermalState thermal = NSProcessInfo.processInfo.thermalState;
        NSArray<NSString *> *labels = @[@"nominal", @"fair", @"serious", @"critical"];
        NSString *label = (thermal >= NSProcessInfoThermalStateNominal && thermal <= NSProcessInfoThermalStateCritical)
            ? labels[(NSUInteger)thermal] : @"unknown";
        NSMutableDictionary *output = [@{@"pid": @(pid), @"thermal_state": label,
                                         @"thermal_raw_value": @(thermal)} mutableCopy];
        struct xsw_usage swap = {0};
        size_t swap_size = sizeof(swap);
        if (sysctlbyname("vm.swapusage", &swap, &swap_size, NULL, 0) == 0 && swap_size == sizeof(swap)) {
            output[@"swap_used_bytes"] = @(swap.xsu_used);
        } else {
            output[@"swap_used_bytes"] = [NSNull null];
        }
        uint64_t physical_memory = 0;
        size_t physical_memory_size = sizeof(physical_memory);
        if (sysctlbyname("hw.memsize", &physical_memory, &physical_memory_size, NULL, 0) == 0 &&
            physical_memory_size == sizeof(physical_memory)) {
            output[@"physical_memory_bytes"] = @(physical_memory);
        } else {
            output[@"physical_memory_bytes"] = [NSNull null];
        }
        if (status == 0) output[@"physical_footprint_bytes"] = @(usage.ri_phys_footprint);
        else output[@"physical_footprint_bytes"] = [NSNull null];
        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:output options:NSJSONWritingSortedKeys error:&error];
        if (data == nil) {
            fprintf(stderr, "cannot encode metrics: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }
        fwrite(data.bytes, 1, data.length, stdout);
        fputc('\n', stdout);
        return 0;
    }
}
