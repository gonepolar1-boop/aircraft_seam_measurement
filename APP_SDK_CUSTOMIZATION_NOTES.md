# App SDK Customization Notes

This repository intentionally excludes the vendor SDK under `app/`.

Reason:
- The local `app/MPSizectorS SDK V2.70` tree is derived from the vendor SDK.
- It contains vendor source, binaries, demos, and generated build artifacts.
- Publishing that tree in a public repository may create licensing or redistribution risk.

What is kept in this repository instead:
- The Python measurement pipeline under `src/`
- Supporting scripts under `scripts/`
- This note describing the local SDK-side integration points

## Local SDK changes identified against `app/MPSizectorS SDK V2.70_init`

The current local SDK tree differs from the original SDK mainly in these files:

- `03_Code/MPSizectorS_ControlCenter/PipelineLaunchDialog.cs`
  - New dialog used to configure and launch the local gap/flush pipeline.
- `03_Code/MPSizectorS_ControlCenter/PipelineResultWindow.cs`
  - New result window used to display pipeline outputs.
- `03_Code/MPSizectorS_ControlCenter/MainFrm.cs`
  - Added menu items and launch flow for the gap/flush pipeline.
  - Added local Python probing, checkpoint selection, point-map export, pipeline process execution, result parsing, and optional 3D viewer launch.
- `03_Code/MPSizectorS_ControlCenter/MainFrm.Designer.cs`
  - UI/layout adjustment for the control-center window.
- `03_Code/MPSizectorS_ControlCenter/31_MPSizectorS_ControlCenter.csproj`
  - Added the new pipeline-related C# files to the project.

## Practical recommendation

If the SDK-side integration needs to be shared later, prefer one of these approaches:

- Re-implement the integration logic in a clean-room project you fully own.
- Share a high-level patch description without redistributing vendor source files.
- Keep the SDK integration in a private repository only after confirming the vendor license permits it.

## Initial public version scope

The intended public initial version of this repository includes:

- `src/`
- `scripts/`
- `.gitignore`
- this note

The following remain local-only and are not part of the public initial version:

- `app/`
- `data/`
- `outputs/`
- local editor and build caches
