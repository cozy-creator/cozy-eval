"""Regenerate tests/fixtures/*.png deterministically. Run from the repo root."""

import numpy as np
from PIL import Image, ImageFilter


def textured(size=384, seed=7):
    """Natural-ish statistics: low-frequency structure + 1/f texture. Flat
    fields and white noise are pathological for NSS metrics; fixtures need
    something in between (plus the pathological extreme, kept on purpose)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size] / size
    base = 96 + 64 * np.sin(2 * np.pi * x * 3) * np.cos(2 * np.pi * y * 2)
    spec = np.fft.rfft2(rng.normal(size=(size, size)))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    pink = np.fft.irfft2(spec / np.maximum(np.hypot(fy, fx), 1.0 / size), s=(size, size))
    pink = pink / np.abs(pink).max() * 48
    return np.clip(base + pink, 0, 255).astype(np.uint8)


def main() -> None:
    rgb = np.stack([textured(seed=s) for s in (7, 8, 9)], axis=-1)
    scene = Image.fromarray(rgb)
    rng = np.random.default_rng(11)
    noisy = np.clip(rgb.astype(np.int16) + rng.normal(0, 25, rgb.shape), 0, 255).astype(np.uint8)
    noise = rng.integers(0, 256, rgb.shape, dtype=np.uint8).astype(np.uint8)
    out = {
        "scene": scene,
        "scene_blur": scene.filter(ImageFilter.GaussianBlur(3)),
        "scene_noise": Image.fromarray(noisy),
        "whitenoise": Image.fromarray(noise),
    }
    for name, im in out.items():
        im.save(f"tests/fixtures/{name}.png")
        print("wrote", name, im.size)


if __name__ == "__main__":
    main()
