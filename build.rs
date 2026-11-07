use std::env;
use std::fs::create_dir;
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
    println!("cargo:rustc-check-cfg=cfg(headless_stub)");

    if env::var("KURP_HEADLESS_TEST").as_deref() == Ok("1") {
        println!("cargo:rustc-cfg=headless_stub");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let glslang_dir = env::var("GLSLANG_TARGET_DIR").ok();
    let realcugan_dir = out_dir.join("realcugan");
    create_dir(&realcugan_dir).unwrap_or_default();
    let realcugan = {
        let mut config = Config::new("src/");
        config.out_dir(realcugan_dir);
        if let Some(dir) = glslang_dir {
            config.define("GLSLANG_TARGET_DIR", dir);
        }
        config.build()
    };
    println!("cargo:rustc-link-search=native={}", realcugan.join("lib").display());
    println!("cargo:rustc-link-lib=static:-bundle={}", "realcugan-ncnn-vulkan-wrapper");
    println!("cargo:rustc-link-lib=dylib=ncnn");
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib={}", "stdc++");
    }
}
