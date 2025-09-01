/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/tsl/platform/path.h"

#include <stdlib.h>

#include "absl/strings/str_cat.h"

#include "xla/tsl/platform/platform.h"
#include "xla/tsl/platform/scanner.h"

namespace tsl::io {
namespace internal {
namespace {

const char kPathSep[] = "/";

// Return the parts of the URI, split on the final "/" in the path. If there is
// no "/" in the path, the first part of the output is the scheme and host, and
// the second is the path. If the only "/" in the path is the first character,
// it is included in the first part of the output.
std::pair<std::string_view, std::string_view> SplitPath(std::string_view uri) {
  std::string_view scheme, host, path;
  ParseURI(uri, &scheme, &host, &path);

  auto pos = path.rfind('/');
#ifdef PLATFORM_WINDOWS
  if (pos == StringPiece::npos) pos = path.rfind('\\');
#endif
  // Handle the case with no '/' in 'path'.
  if (pos == std::string_view::npos)
    return std::make_pair(
        std::string_view(uri.data(), host.end() - uri.begin()), path);

  // Handle the case with a single leading '/' in 'path'.
  if (pos == 0)
    return std::make_pair(
        std::string_view(uri.data(), path.begin() + 1 - uri.begin()),
        std::string_view(path.data() + 1, path.size() - 1));

  return std::make_pair(
      std::string_view(uri.data(), path.begin() + pos - uri.begin()),
      std::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}

// Return the parts of the basename of path, split on the final ".".
// If there is no "." in the basename or "." is the final character in the
// basename, the second value will be empty.
std::pair<std::string_view, std::string_view> SplitBasename(
    std::string_view path) {
  path = Basename(path);

  auto pos = path.rfind('.');
  if (pos == std::string_view::npos)
    return std::make_pair(path, std::string_view(path.data() + path.size(), 0));
  return std::make_pair(
      std::string_view(path.data(), pos),
      std::string_view(path.data() + pos + 1, path.size() - (pos + 1)));
}

}  // namespace

std::string JoinPathImpl(std::initializer_list<std::string_view> paths) {
  std::string result;

  for (std::string_view path : paths) {
    if (path.empty()) continue;

    if (result.empty()) {
      result = std::string(path);
      continue;
    }

    if (IsAbsolutePath(path)) path = path.substr(1);

    if (result[result.size() - 1] == kPathSep[0]) {
      absl::StrAppend(&result, path);
    } else {
      absl::StrAppend(&result, kPathSep, path);
    }
  }

  return result;
}

}  // namespace internal

bool IsAbsolutePath(std::string_view path) {
  return !path.empty() && path[0] == '/';
}

std::string_view Dirname(std::string_view path) {
  return internal::SplitPath(path).first;
}

std::string_view Basename(std::string_view path) {
  return internal::SplitPath(path).second;
}

std::string_view Extension(std::string_view path) {
  return internal::SplitBasename(path).second;
}

std::string_view BasenamePrefix(std::string_view path) {
  return internal::SplitBasename(path).first;
}

std::string CleanPath(std::string_view unclean_path) {
  std::string path(unclean_path);
  const char* src = path.c_str();
  std::string::iterator dst = path.begin();

  // Check for absolute path and determine initial backtrack limit.
  const bool is_absolute_path = *src == '/';
  if (is_absolute_path) {
    *dst++ = *src++;
    while (*src == '/') ++src;
  }
  std::string::const_iterator backtrack_limit = dst;

  // Process all parts
  while (*src) {
    bool parsed = false;

    if (src[0] == '.') {
      //  1dot ".<whateverisnext>", check for END or SEP.
      if (src[1] == '/' || !src[1]) {
        if (*++src) {
          ++src;
        }
        parsed = true;
      } else if (src[1] == '.' && (src[2] == '/' || !src[2])) {
        // 2dot END or SEP (".." | "../<whateverisnext>").
        src += 2;
        if (dst != backtrack_limit) {
          // We can backtrack the previous part
          for (--dst; dst != backtrack_limit && dst[-1] != '/'; --dst) {
            // Empty.
          }
        } else if (!is_absolute_path) {
          // Failed to backtrack and we can't skip it either. Rewind and copy.
          src -= 2;
          *dst++ = *src++;
          *dst++ = *src++;
          if (*src) {
            *dst++ = *src;
          }
          // We can never backtrack over a copied "../" part so set new limit.
          backtrack_limit = dst;
        }
        if (*src) {
          ++src;
        }
        parsed = true;
      }
    }

    // If not parsed, copy entire part until the next SEP or EOS.
    if (!parsed) {
      while (*src && *src != '/') {
        *dst++ = *src++;
      }
      if (*src) {
        *dst++ = *src++;
      }
    }

    // Skip consecutive SEP occurrences
    while (*src == '/') {
      ++src;
    }
  }

  // Calculate and check the length of the cleaned path.
  std::string::difference_type path_length = dst - path.begin();
  if (path_length != 0) {
    // Remove trailing '/' except if it is root path ("/" ==> path_length := 1)
    if (path_length > 1 && path[path_length - 1] == '/') {
      --path_length;
    }
    path.resize(path_length);
  } else {
    // The cleaned path is empty; assign "." as per the spec.
    path.assign(1, '.');
  }
  return path;
}

void ParseURI(std::string_view uri, std::string_view* scheme,
              std::string_view* host, std::string_view* path) {
  // 0. Parse scheme
  // Make sure scheme matches [a-zA-Z][0-9a-zA-Z.]*
  // TODO(keveman): Allow "+" and "-" in the scheme.
  // Keep URI pattern in TensorBoard's `_parse_event_files_spec` updated
  // accordingly
  if (!strings::Scanner(uri)
           .One(strings::Scanner::LETTER)
           .Many(strings::Scanner::LETTER_DIGIT_DOT)
           .StopCapture()
           .OneLiteral("://")
           .GetResult(&uri, scheme)) {
    // If there's no scheme, assume the entire string is a path.
    *scheme = std::string_view(uri.data(), 0);
    *host = std::string_view(uri.data(), 0);
    *path = uri;
    return;
  }

  // 1. Parse host
  if (!strings::Scanner(uri).ScanUntil('/').GetResult(&uri, host)) {
    // No path, so the rest of the URI is the host.
    *host = uri;
    *path = std::string_view();  // empty path
    return;
  }

  // 2. The rest is the path
  *path = uri;
}

std::string CreateURI(std::string_view scheme, std::string_view host,
                      std::string_view path) {
  if (scheme.empty()) {
    return std::string(path);
  }
  return absl::StrCat(scheme, "://", host, path);
}

bool GetTestWorkspaceDir(std::string* dir) {
  const char* srcdir = getenv("TEST_SRCDIR");
  if (srcdir == nullptr) {
    return false;
  }
  const char* workspace = getenv("TEST_WORKSPACE");
  if (workspace == nullptr) {
    return false;
  }
  if (dir != nullptr) {
    *dir = tsl::io::JoinPath(srcdir, workspace);
  }
  return true;
}

bool GetTestUndeclaredOutputsDir(std::string* dir) {
  const char* outputs_dir = getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (outputs_dir == nullptr) {
    return false;
  }
  if (dir != nullptr) {
    *dir = outputs_dir;
  }
  return true;
}

[[maybe_unused]] std::string& AppendDotExeIfWindows(std::string& path) {
#ifdef PLATFORM_WINDOWS
  path.append(".exe");
#endif  // PLATFORM_WINDOWS
  return path;
}

}  // namespace tsl::io
