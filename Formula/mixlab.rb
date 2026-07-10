class Mixlab < Formula
  desc "ML architecture exploration tool — JSON configs, Go IR, Metal/CUDA"
  homepage "https://github.com/mrothroc/mixlab"
  url "https://github.com/mrothroc/mixlab.git",
      tag:      "v0.57.0",
      revision: "727897b495b8a2c7774c105f93cbb7b3801d2de9"
  license "MIT"
  head "https://github.com/mrothroc/mixlab.git", branch: "main"

  depends_on "go" => :build
  depends_on "mlx"
  depends_on :macos

  def install
    mlx_prefix = Formula["mlx"].opt_prefix

    ENV["CGO_ENABLED"] = "1"
    ENV.append "CGO_CFLAGS", "-I#{mlx_prefix}/include"
    ENV.append "CGO_CXXFLAGS", "-I#{mlx_prefix}/include -std=c++20"
    ENV.append "CGO_LDFLAGS", "-L#{mlx_prefix}/lib -Wl,-rpath,#{mlx_prefix}/lib"

    system "go", "build", "-tags", "mlx",
           "-o", bin/"mixlab", "./cmd/mixlab"
  end

  test do
    assert_match "PASS", shell_output("#{bin}/mixlab -mode smoke 2>&1")
  end
end
