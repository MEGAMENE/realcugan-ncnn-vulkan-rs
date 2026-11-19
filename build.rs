use std::env;
use std::fs;

use cmake::Config;

fn main() {
    println!("cargo:rerun-if-changed=src/realcugan_wrapped.cpp");
    println!("cargo:rerun-if-changed=src/CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan.cpp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan.h");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/generate_shader_comp_header.cmake");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_preproc.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_preproc_tta.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_postproc.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_postproc_tta.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_4x_postproc.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_4x_postproc_tta.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/ncnn/CMakeLists.txt");
    println!("cargo:rerun-if-env-changed=REALCUGAN_NATIVE");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let crt_static = target_features.split(',').any(|feat| feat == "crt-static");

    let mut config = Config::new("src");
    config.profile("Release");

    if target_env == "msvc" {
        // Disable MSBuild file tracker via environment property to avoid MAX_PATH FTK1011 errors
        // without passing MSBuild-only CLI arguments that break Ninja/Make generators.
        config.env("TrackFileAccess", "false");
        let msvc_rt = if crt_static {
            "MultiThreaded"
        } else {
            "MultiThreadedDLL"
        };
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", msvc_rt);
    }

    if let Ok(native) = env::var("REALCUGAN_NATIVE") {
        if matches!(native.as_str(), "1" | "ON" | "on" | "TRUE" | "true") {
            config.define("REALCUGAN_NATIVE", "ON");
        }
    }

    let dst = config.build();

    let manifest_path = dst.join("link_manifest.txt");
    let manifest = fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
        panic!(
            "failed to read CMake link manifest at {}: {}",
            manifest_path.display(),
            err
        )
    });

    for line in manifest.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let Some((key, val)) = line.split_once('=') else {
            panic!("malformed link_manifest.txt entry: {line}");
        };
        match key {
            "LIB_SUBDIR" => {
                println!("cargo:rustc-link-search=native={}", dst.join(val).display());
            }
            "SEARCH_DIR" => {
                println!("cargo:rustc-link-search=native={val}");
            }
            "STATIC_LIB" => {
                println!("cargo:rustc-link-lib=static={val}");
            }
            "DYLIB" => {
                println!("cargo:rustc-link-lib=dylib={val}");
            }
            "FRAMEWORK" => {
                println!("cargo:rustc-link-lib=framework={val}");
            }
            "LINK_ARG" => {
                println!("cargo:rustc-link-arg={val}");
            }
            other => {
                panic!("unknown link_manifest.txt key: {other}");
            }
        }
    }

    match target_os.as_str() {
        "macos" | "ios" | "freebsd" | "openbsd" => {
            println!("cargo:rustc-link-lib=dylib=c++");
        }
        "linux" | "android" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        "windows" if target_env == "gnu" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        _ => {}
    }
}
