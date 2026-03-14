# Clang Tooling

This project ships repository-level `clang-format` and `clang-tidy` configuration files:

- `.clang-format`
- `.clang-tidy`

They are designed for modern C++20 code and aligned with this repository's conventions.

## Style Rules Enforced

### clang-format

- 4-space indentation.
- Attached braces (`if (...) {`).
- 120-column line limit.
- Pointer/reference alignment on the left (`Type* ptr`, `Type& ref`).
- Stable include sorting while preserving include blocks.

### clang-tidy

- Safety and correctness checks: `bugprone-*`, `portability-*`.
- Performance checks: `performance-*`.
- Modernization checks: `modernize-*`.
- Readability checks: `readability-*`.
- Naming conventions:
  - `PascalCase` for classes/structs/enums/type aliases.
  - `lower_case` for functions, methods, variables, parameters, and members.
  - `kPascalCase` for `constexpr` and global constants.
  - Private member suffix `_`.

## Local Usage (PowerShell)

### 1) Generate compile commands

Use a Ninja build directory dedicated to analysis:

```powershell
cmake -S . -B build-clang -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### 2) Run clang-format

```powershell
Get-ChildItem include,src -Recurse -File -Include *.hpp,*.cpp |
    ForEach-Object { clang-format -i $_.FullName }
```

### 3) Run clang-tidy for all translation units

```powershell
Get-ChildItem src -Recurse -File -Filter *.cpp |
    ForEach-Object { clang-tidy $_.FullName -p build-clang --config-file=.clang-tidy --quiet }
```

## Notes

- `clang-tidy` needs a valid `compile_commands.json`.
- Run formatting before static analysis to reduce noisy diagnostics.
- Start with warnings, then tighten policy (for example, `WarningsAsErrors`) once the baseline is clean.
