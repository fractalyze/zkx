// Copyright (c) 2020 The Console Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ZKX_BASE_FLAG_FLAG_PARSER_H_
#define ZKX_BASE_FLAG_FLAG_PARSER_H_

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"

#include "zkx/base/flag/flag.h"

namespace zkx::base {

class FlagParserBase {
 public:
  FlagParserBase();
  FlagParserBase(const FlagParserBase& other) = delete;
  FlagParserBase& operator=(const FlagParserBase& other) = delete;
  virtual ~FlagParserBase();

  const std::vector<std::unique_ptr<FlagBase>>* flags() const {
    return &flags_;
  }

  template <typename T, typename value_type = typename Flag<T>::value_type>
  T& AddFlag(value_type* value) {
    std::unique_ptr<FlagBase> flag(new T(value));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T, typename value_type = typename Flag<T>::value_type,
            typename ParseValueCallback = typename Flag<T>::ParseValueCallback>
  T& AddFlag(value_type* value, ParseValueCallback parse_value_callback) {
    std::unique_ptr<FlagBase> flag(new T(value, parse_value_callback));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T,
            typename value_type = typename ChoicesFlag<T>::value_type>
  T& AddFlag(value_type* value, const std::vector<value_type>& choices) {
    std::unique_ptr<FlagBase> flag(new T(value, choices));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T,
            typename value_type = typename ChoicesFlag<T>::value_type>
  T& AddFlag(value_type* value, std::vector<value_type>&& choices) {
    std::unique_ptr<FlagBase> flag(new T(value, std::move(choices)));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  template <typename T, typename value_type = typename RangeFlag<T>::value_type>
  T& AddFlag(value_type* value, const value_type& start,
             const value_type& end) {
    std::unique_ptr<FlagBase> flag(new T(value, start, end));
    flags_.push_back(std::move(flag));
    return *reinterpret_cast<T*>(flags_.back().get());
  }

  FlagBase& AddFlag(std::unique_ptr<FlagBase> flag_base);

  SubParser& AddSubParser();

 protected:
  FRIEND_TEST(FlagParserTest, ValidateInternally);

  struct Context {
    FlagParser* parser;
    int current_idx;
    int argc;
    char** argv;
    int* unknown_argc = nullptr;
    std::vector<char*> unknown_argv;
    std::vector<std::string>* forward_argv = nullptr;

    Context(FlagParser* parser, int current_idx, int argc, char** argv);
    ~Context();

    std::string_view current() const;
    bool ConsumeEqualOrProceed(std::string_view* arg);
    void Proceed();
    bool HasArg() const;
    void FillUnknownArgs() const;
  };

  absl::Status Parse(Context& ctx);

  absl::Status ValidateInternally() const;

  // Internally it measures Levenshtein distance among arguments.
  absl::StatusOr<std::string_view> FindTheMostSimilarFlag(
      std::string_view input);

 protected:
  std::vector<std::unique_ptr<FlagBase>> flags_;
};

class FlagParser : public FlagParserBase {
 public:
  FlagParser();
  FlagParser(const FlagParser& other) = delete;
  FlagParser& operator=(const FlagParser& other) = delete;
  ~FlagParser();

  void set_program_path(std::string_view program_path) {
    program_path_ = std::string(program_path);
  }
  const std::string& program_path() const { return program_path_; }

  // Sames as ParseWithForward(argc, argv, nullptr).
  absl::Status Parse(int argc, char** argv);

  // Sames as ParseKnownWithForward(argc, argv, nullptr).
  // It only parses known arguments. After parsing, `argc` and `argv` are
  // updated to match the unknown arguments.
  absl::Status ParseKnown(int* argc, char** argv);

  absl::Status ParseWithForward(int argc, char** argv,
                                std::vector<std::string>* forward);

  absl::Status ParseKnownWithForward(int* argc, char** argv,
                                     std::vector<std::string>* forward);

  // It is marked virtual so that users can make custom help messages.
  virtual std::string help_message();

  virtual absl::Status Validate() { return absl::OkStatus(); }

 private:
  absl::Status PreParse(int argc, char** argv);

  std::string program_path_;
};

class SubParser : public FlagBase,
                  public FlagParserBase,
                  public FlagBaseBuilder<SubParser> {
 public:
  SubParser();
  ~SubParser();
  SubParser(const SubParser& other) = delete;
  SubParser& operator=(const SubParser& other) = delete;

  // FlagBase methods
  bool IsSubParser() const override;
  bool NeedsValue() const override;
  absl::Status ParseValue(std::string_view arg) override;

  SubParser& set_is_set(bool* is_set) {
    is_set_ = is_set;
    return *this;
  }

 private:
  friend class FlagParserBase;

  bool* is_set_ = nullptr;
};

}  // namespace zkx::base

#endif  // ZKX_BASE_FLAG_FLAG_PARSER_H_
