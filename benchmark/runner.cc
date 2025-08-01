#include "benchmark/runner.h"

#include "zkx/base/console/table_writer.h"
#include "zkx/base/csv/csv_writer.h"
#include "zkx/base/flag/numeric_flags.h"
#include "zkx/service/platform_util.h"

namespace zkx::benchmark {

Runner::Runner() : runner_(PlatformUtil::GetPlatform("cpu").value()) {}

void Runner::AddPositionalFlags() {
  parser_.AddFlag<base::StringFlag>(&input_path_)
      .set_name("input")
      .set_help("Input file");
}

void Runner::AddOptionalFlags(const Options& options) {
  if (options.multi_degrees) {
    parser_.AddFlag<base::Flag<std::vector<uint32_t>>>(&degrees_)
        .set_short_name("-k")
        .set_required()
        .set_help("Degree of the inputs. Can be specified multiple times.");
  }

  parser_.AddFlag<base::Uint32Flag>(&num_warmups_)
      .set_long_name("--num_warmups")
      .set_default_value(0)
      .set_help("Number of warmups. Default: 0");
  parser_
      .AddFlag<base::Uint32Flag>(&num_runs_,
                                 &base::ParsePositiveValue<uint32_t>)
      .set_long_name("--num_runs")
      .set_default_value(1)
      .set_help("Number of runs. Default: 1");
  parser_.AddFlag<base::StringFlag>(&output_path_)
      .set_short_name("-o")
      .set_long_name("--output")
      .set_help("Output file (.csv)");
}

absl::Status Runner::Parse(int argc, char** argv) {
  TF_RETURN_IF_ERROR(parser_.Parse(argc, argv));
  std::sort(degrees_.begin(), degrees_.end());
  return absl::OkStatus();
}

void Runner::PrintResults() const {
  base::TableWriterBuilder builder;
  builder.AlignHeaderLeft()
      .AddSpace(1)
      .FitToTerminalWidth()
      .StripTrailingAsciiWhitespace()
      .AddColumn("degree");

  for (uint32_t i = 0; i < num_runs_; ++i) {
    builder.AddColumn(absl::StrCat("run", i + 1, " (ms)"));
  }
  builder.AddColumn("avg (ms)");

  base::TableWriter writer = builder.Build();

  for (size_t i = 0; i < degrees_.size(); ++i) {
    writer.SetElement(i, 0, absl::StrCat(degrees_[i]));
    absl::Duration total_duration = absl::ZeroDuration();
    for (uint32_t j = 0; j < num_runs_; ++j) {
      writer.SetElement(
          i, j + 1, absl::StrCat(absl::ToDoubleMilliseconds(durations_[i][j])));
      total_duration += durations_[i][j];
    }
    writer.SetElement(
        i, num_runs_ + 1,
        absl::StrCat(absl::ToDoubleMilliseconds(total_duration / num_runs_)));
  }
  writer.Print(true);
}

absl::Status Runner::MaybeWriteCsv() const {
  if (output_path_.empty()) {
    return absl::OkStatus();
  }

  base::CsvWriter writer;
  std::vector<std::string> titles;
  titles.push_back("degree");
  for (uint32_t i = 0; i < num_runs_; ++i) {
    titles.push_back(absl::StrCat("run", i + 1));
  }
  writer.WriteRow(titles);
  for (size_t i = 0; i < degrees_.size(); ++i) {
    std::vector<std::string> row;
    row.push_back(absl::StrCat(degrees_[i]));
    for (uint32_t j = 0; j < num_runs_; ++j) {
      row.push_back(absl::StrCat(absl::ToDoubleMilliseconds(durations_[i][j])));
    }
    writer.WriteRow(row);
  }

  return writer.WriteToFile(output_path_);
}

}  // namespace zkx::benchmark
