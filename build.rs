use std::env;
use std::fs;
use std::path::PathBuf;

use cmake::Config;

fn main() {
    println!("cargo:rerun-if-changed=src/realcugan_wrapped.cpp");
    println!("cargo:rerun-if-changed=src/CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan.cpp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan.h");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_preproc.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_preproc_tta.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_postproc.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_postproc_tta.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_4x_postproc.comp");
    println!("cargo:rerun-if-changed=src/realcugan-ncnn-vulkan/src/realcugan_4x_postproc_tta.comp");
    println!("cargo:rerun-if-env-changed=KURP_HEADLESS_TEST");
    println!("cargo:rerun-if-env-changed=NCNN_DIR");
    println!("cargo:rustc-check-cfg=cfg(headless_stub)");

    if env::var("KURP_HEADLESS_TEST").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=headless_stub");
        return;
    }

    // Disable MSBuild file tracker to avoid MAX_PATH failure on Windows deep build paths
    std::env::set_var("TrackFileAccess", "false");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let ncnn_dir = env::var("NCNN_DIR")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let candidate = PathBuf::from(
                r"C:\Users\Courier\Documents\Antigravity Projects\Kurp Matrix Workbench\ncnn\ncnn-20260526-windows-vs2022-shared\x64\lib\cmake\ncnn",
            );
            if candidate.exists() {
                Some(candidate)
            } else {
                None
            }
        });

    let realcugan = {
        let mut config = Config::new("src/");
        config.profile("Release");
        if let Some(ref dir) = ncnn_dir {
            config.define("ncnn_DIR", dir);
            config.define("USE_SYSTEM_NCNN", "ON");
        }
        config.build_arg("/p:TrackFileAccess=false");
        config.build()
    };

    println!("cargo:rustc-link-search=native={}", realcugan.join("lib").display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-ncnn-vulkan-wrapper");

    if let Some(ref dir) = ncnn_dir {
        // .../lib/cmake/ncnn -> .../lib -> .../bin
        if let Some(cmake_dir) = dir.parent() {
            if let Some(lib_dir) = cmake_dir.parent() {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                if let Some(root_dir) = lib_dir.parent() {
                    let bin_dir = root_dir.join("bin");
                    if bin_dir.exists() {
                        println!("cargo:rustc-link-search=native={}", bin_dir.display());
                        let ncnn_dll = bin_dir.join("ncnn.dll");
                        if ncnn_dll.exists() {
                            let _ = fs::copy(&ncnn_dll, out_dir.join("ncnn.dll"));
                            // Copy to target directory hierarchy so cargo test / run find ncnn.dll
                            if let Some(target_dir) = out_dir.ancestors().nth(3) {
                                let _ = fs::copy(&ncnn_dll, target_dir.join("ncnn.dll"));
                                let _ = fs::copy(&ncnn_dll, target_dir.join("deps").join("ncnn.dll"));
                                let _ = fs::copy(&ncnn_dll, target_dir.join("examples").join("ncnn.dll"));
                            }
                        }
                    }
                }
            }
        }
    }

    println!("cargo:rustc-link-lib=dylib=ncnn");
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib={}", "stdc++");
    }
}
