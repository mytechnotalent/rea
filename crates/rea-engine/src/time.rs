//! Timestamp Formatting
//!
//! **File:** `time.rs`
//! **Author:** Kevin Thomas
//! **Date:** 2026
//!
//! MIT License
//!
//! Copyright (c) 2026 Kevin Thomas
//!
//! Permission is hereby granted, free of charge, to any person obtaining a copy
//! of this software and associated documentation files (the "Software"), to deal
//! in the Software without restriction, including without limitation the rights
//! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! copies of the Software, and to permit persons to whom the Software is
//! furnished to do so, subject to the following conditions:
//!
//! The above copyright notice and this permission notice shall be included in
//! all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! SOFTWARE.

/// Seconds in one day.
const DAY: u64 = 86_400;

/// Formats a Unix timestamp as a UTC date and time.
///
/// # Arguments
///
/// * `seconds` - Seconds since 1970-01-01T00:00:00Z.
///
/// # Returns
///
/// Text such as `2024-03-01 12:00:00 UTC`.
#[must_use]
pub fn utc_timestamp(seconds: u64) -> String {
    let (days, rest) = (seconds / DAY, seconds % DAY);
    let (year, month, day) = civil_date(days);
    let (hour, minute, second) = (rest / 3600, rest % 3600 / 60, rest % 60);
    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

/// Converts days since the Unix epoch to a proleptic Gregorian date.
///
/// # Arguments
///
/// * `days` - Days since 1970-01-01.
///
/// # Returns
///
/// The `(year, month, day)` triple.
fn civil_date(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let (era, doe) = (z / 146_097, z % 146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let (day, month) = (doy - (153 * mp + 2) / 5 + 1, (mp + 2) % 12 + 1);
    (yoe + era * 400 + u64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    // Import all parent module items
    use super::*;

    /// Verifies the epoch itself.
    #[test]
    fn test_utc_timestamp_epoch() {
        assert_eq!(utc_timestamp(0), "1970-01-01 00:00:00 UTC");
    }

    /// Verifies a leap day and a time of day.
    #[test]
    fn test_utc_timestamp_leap_day() {
        assert_eq!(utc_timestamp(951_827_696), "2000-02-29 12:34:56 UTC");
    }

    /// Verifies year rollover at the end of December.
    #[test]
    fn test_civil_date_year_end() {
        assert_eq!(civil_date(18_627), (2020, 12, 31));
        assert_eq!(civil_date(18_628), (2021, 1, 1));
    }
}
