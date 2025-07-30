# Avalanche Simulation with WebGPU

[Try it yourself!](https://gro2mi.github.io/AvalancheSim-WebGPU/ "Avalanche Simulation") This only works in Chrome (or other chromium based browsers as of June 2025) and Firefox >=141 on Windows. You might have to enable WebGPU flags at `chrome://flags`. It was tested on Windows, Linux and Android but support on Mobile might be lacking.

This project is to improve the development process for avalanche simulations with webGPU based on [weBIGeo](https://github.com/weBIGeo/webigeo/tree/main). It offers the possibility to easily plot results in the browser.

Test examples are from [AvaFrame
](https://docs.avaframe.org/en/latest/testing.html#tests-for-model-validation)

Tiles are provided by the [AlpineMaps project](https://github.com/AlpineMapsOrg)

Requirements: Python (or Webserver), Browser with [WebGPU support](https://caniuse.com/webgpu) (currently only Chromium based browsers. You might have to enable WebGPU flags at `chrome://flags`. And since July 2025 Firefox on Windows)

1. Go to this directory
2. Start server with `python .\dev_server.py` for disabled cache and a secure connection with self signed certs which are needed to use WebGPU (except for localhost where `python -m http.server 8000` works as well)
3. Open Chrome on [https://localhost/index.html](https://localhost/index.html) or [https://localhost/index.html?debug=vscode](https://localhost/index.html?debug=vscode) for debugging mode or replace localhost with IP if accessing from another device.

## Known Issues

* Chromium on Windows currently ignores the `high-performance` option in `powerPreference` WebGPU flag [[Issue](https://crbug.com/369219127)]. Options are:
  1. Run slow on integrated GPU
  2. Start Chrome with high performance gpu flag `"C:\Program Files\Google\Chrome\Application\chrome.exe" --force_high_performance_gpu`
  3. Activate flag in Chrome `chrome://flags/#force-high-performance-gpu`. I still get slow runs about 50% of the time even though it actually runs on the fast GPU
  4. Activate the dedicated GPU for Chrome in the system settings.
