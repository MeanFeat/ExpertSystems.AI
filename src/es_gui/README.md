# es_gui – ImGui trainer

Boilerplate Dear ImGui front end for ExpertSystems.AI.

* `es_profile.h` – `.esprofile` text format bundling network, train/test data and hyper-parameters
  (relative paths, plain `key=value`, trivially read/written from UE5 C++/Python, e.g. from the ProcAnim plugin).
* `es_async_trainer.h` – worker-thread training; the UI never blocks and can stop/save at any time.
* `es_gui_app.cpp` – Win32 + DirectX11 ImGui app: new/load/save network, load profile, start/stop training,
  train and test cost histograms.

## Building
Add `es_gui_app.cpp` to a new Win32 app project that references `es_core`, and add the Dear ImGui sources
(`imgui*.cpp`, `backends/imgui_impl_win32.cpp`, `backends/imgui_impl_dx11.cpp`) from https://github.com/ocornut/imgui.

Data CSVs are loaded with `Eigen::BuildMatFromFile` (one CSV line per matrix row; features x samples, labels x samples).

## UE5 round trip
Export data from UE as CSV (`train_x.csv`, ...) next to a `.esprofile`, open the profile here, train, save the
network to the path in the profile; UE imports the network file (same JSON-like layer format as `Net::SaveNetwork`).
