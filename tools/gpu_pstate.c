// GPU performance-state residency and frequency of an Apple-silicon Mac, without root (IOReport, as asitop/macmon).
// Prints one line per interval: time, average GPU frequency over the busy time, busy %, and residency per state.
//
//   clang -O2 tools/gpu_pstate.c -o /tmp/gpu_pstate -framework CoreFoundation -framework IOKit -lIOReport
//   /tmp/gpu_pstate [interval_ms=500] [count=0 (forever)]
//
// The state frequencies come from the "voltage-states9" property of the pmgr node (M1/M2 layout).
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef struct IOReportSubscriptionRef * IOReportSubscriptionRef;
extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(CFStringRef group, CFStringRef subgroup, uint64_t a, uint64_t b, uint64_t c);
extern IOReportSubscriptionRef IOReportCreateSubscription(void * a, CFMutableDictionaryRef chans, CFMutableDictionaryRef * subbed, uint64_t b, CFTypeRef c);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef sub, CFMutableDictionaryRef chans, CFTypeRef a);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef a, CFDictionaryRef b, CFTypeRef c);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef ch);
extern int IOReportStateGetCount(CFDictionaryRef ch);
extern CFStringRef IOReportStateGetNameForIndex(CFDictionaryRef ch, int i);
extern int64_t IOReportStateGetResidency(CFDictionaryRef ch, int i);

static int gpu_freqs_mhz(double * out, int max) {
    // pmgr exposes the GPU DVFS table as pairs of uint32 (freq Hz, voltage)
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceNameMatching("pmgr"), &it) != KERN_SUCCESS) return 0;
    int n = 0;
    io_object_t e;
    while ((e = IOIteratorNext(it))) {
        CFDataRef d = IORegistryEntryCreateCFProperty(e, CFSTR("voltage-states9"), kCFAllocatorDefault, 0);
        if (d) {
            const uint32_t * p = (const uint32_t *) CFDataGetBytePtr(d);
            int len = (int) CFDataGetLength(d) / 8;
            for (int i = 0; i < len && n < max; ++i) if (p[2*i]) out[n++] = p[2*i] / 1e6;
            CFRelease(d);
        }
        IOObjectRelease(e);
    }
    IOObjectRelease(it);
    return n;
}

int main(int argc, char ** argv) {
    const int interval = argc > 1 ? atoi(argv[1]) : 500;
    const int count    = argc > 2 ? atoi(argv[2]) : 0;
    double freqs[64];
    const int nf = gpu_freqs_mhz(freqs, 64);

    CFMutableDictionaryRef chans = IOReportCopyChannelsInGroup(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0);
    CFMutableDictionaryRef subbed = NULL;
    IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, chans, &subbed, 0, NULL);
    if (!sub) { fprintf(stderr, "IOReport subscription failed\n"); return 1; }

    CFDictionaryRef prev = IOReportCreateSamples(sub, subbed, NULL);
    for (int k = 0; count == 0 || k < count; ++k) {
        usleep(interval * 1000);
        CFDictionaryRef cur = IOReportCreateSamples(sub, subbed, NULL);
        CFDictionaryRef delta = IOReportCreateSamplesDelta(prev, cur, NULL);
        CFRelease(prev);
        prev = cur;

        CFArrayRef arr = CFDictionaryGetValue(delta, CFSTR("IOReportChannels"));
        for (CFIndex i = 0; arr && i < CFArrayGetCount(arr); ++i) {
            CFDictionaryRef ch = CFArrayGetValueAtIndex(arr, i);
            char name[64] = {0};
            CFStringGetCString(IOReportChannelGetChannelName(ch), name, sizeof(name), kCFStringEncodingUTF8);
            if (strcmp(name, "GPUPH") != 0) continue;   // the GPU P-state residency channel
            const int ns = IOReportStateGetCount(ch);
            int64_t res[64] = {0}, total = 0, busy = 0;
            double fsum = 0;
            int ip = 0;  // index over the active (P) states
            char line[1024]; int off = 0;
            for (int s = 0; s < ns && s < 64; ++s) {
                char sn[32] = {0};
                CFStringGetCString(IOReportStateGetNameForIndex(ch, s), sn, sizeof(sn), kCFStringEncodingUTF8);
                res[s] = IOReportStateGetResidency(ch, s);
                total += res[s];
                if (strcmp(sn, "IDLE") != 0 && strcmp(sn, "OFF") != 0 && strcmp(sn, "DOWN") != 0) {
                    const double f = ip < nf ? freqs[ip] : 0;
                    busy += res[s]; fsum += f * res[s]; ++ip;
                }
            }
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
            struct tm tm; localtime_r(&ts.tv_sec, &tm);
            off += snprintf(line + off, sizeof(line) - off, "%02d:%02d:%02d.%03ld freq %4.0f MHz busy %5.1f%% |",
                            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000,
                            busy ? fsum / busy : 0.0, total ? 100.0 * busy / total : 0.0);
            ip = 0;
            for (int s = 0; s < ns && s < 64; ++s) {
                char sn[32] = {0};
                CFStringGetCString(IOReportStateGetNameForIndex(ch, s), sn, sizeof(sn), kCFStringEncodingUTF8);
                if (strcmp(sn, "IDLE") == 0 || strcmp(sn, "OFF") == 0 || strcmp(sn, "DOWN") == 0) continue;
                if (busy && res[s] * 100 / busy >= 1)
                    off += snprintf(line + off, sizeof(line) - off, " %.0f:%lld%%", ip < nf ? freqs[ip] : 0.0, (long long) (res[s] * 100 / busy));
                ++ip;
            }
            puts(line);
            fflush(stdout);
        }
        CFRelease(delta);
    }
    return 0;
}
