#import <Foundation/Foundation.h>
#import <libproc.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
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
