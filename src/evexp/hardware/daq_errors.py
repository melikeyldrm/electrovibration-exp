"""Friendly wrappers around nidaqmx.DaqError for common failure modes."""

from typing import Optional


class DaqCommunicationError(RuntimeError):
    """Base class for every translated NI-DAQmx error."""


class DaqDeviceNotFoundError(DaqCommunicationError):
    """The device is not present - unplugged, powered off, or never connected."""


class DaqDeviceBusyError(DaqCommunicationError):
    """The device is already reserved by another task or program."""


class DaqBufferOverflowError(DaqCommunicationError):
    """The DAQmx buffer overflowed - the reader fell behind the sample clock."""


class DaqTimeoutError(DaqCommunicationError):
    """A read or write call did not complete within its timeout."""


class DaqChannelError(DaqCommunicationError):
    """A channel name or specification is invalid for this device."""

# NI-DAQmx error_code -> (exception class, human-readable explanation).
_KNOWN_CODES = {
    -50300: (DaqDeviceNotFoundError,
             "device not found - check the USB/PCIe connection and that it "
             "still appears in NI MAX"),
    -200022: (DaqDeviceBusyError,
              "device is already reserved by another task or program (e.g. "
              "an NI MAX test panel left open, or a previous run that "
              "didn't close its task cleanly)"),
    -200279: (DaqBufferOverflowError,
              "buffer overflow (samples no longer available) - the read "
              "loop fell behind the sample clock; lower sample_rate_hz, "
              "raise buffer_seconds, or look for blocking calls in the "
              "read loop"),
    -200221: (DaqTimeoutError,
              "call timed out - no new samples arrived in time; check the "
              "device is running and the sample clock is configured"),
    -200170: (DaqChannelError,
              "physical channel does not exist - check channel_map against "
              "the device's actual channel names in NI MAX"),
    -200087: (DaqChannelError,
              "invalid channel specification - check channel_map syntax"),
}

def translate_daq_error(exc: Exception,
                         device: Optional[str] = None) -> DaqCommunicationError:
    """Wrap a nidaqmx.DaqError in a specific, actionable exception."""
    code = getattr(exc, "error_code", None)
    entry = _KNOWN_CODES.get(code)
    device_note = f" ({device})" if device else ""

    if entry is None:
        return DaqCommunicationError(f"NI-DAQmx error{device_note}: {exc}")

    cls, explanation = entry
    return cls(f"DAQ{device_note}: {explanation}\n\n(NI-DAQmx code {code}: {exc})")