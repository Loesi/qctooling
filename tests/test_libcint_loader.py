import qctooling._libcint as libcint


def test_registers_bundled_and_linked_dll_directories(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "_libcint"
    bundled_dir.mkdir()
    linked_dir = tmp_path / "linked"
    linked_dir.mkdir()
    other_linked_dir = tmp_path / "other-linked"
    other_linked_dir.mkdir()
    handles = [object(), object(), object()]
    add_dll_directory_calls = []

    def add_dll_directory(path):
        add_dll_directory_calls.append(path)
        return handles[len(add_dll_directory_calls) - 1]

    monkeypatch.setattr(libcint.sys, "platform", "win32")
    monkeypatch.setattr(libcint.importlib.resources, "files", lambda package: tmp_path)
    monkeypatch.setattr(libcint.os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setenv(
        "QCTOOLING_LINKED_LIBARY_PATHS",
        f"{linked_dir}{libcint.os.pathsep}{other_linked_dir}",
    )
    libcint._DLL_DIRECTORY_HANDLES.clear()

    libcint._register_linked_libary_paths()

    assert add_dll_directory_calls == [str(bundled_dir), str(linked_dir), str(other_linked_dir)]
    assert libcint._DLL_DIRECTORY_HANDLES == handles


def test_registers_bundled_directory_without_environment_variable(monkeypatch, tmp_path):
    bundled_dir = tmp_path / "_libcint"
    bundled_dir.mkdir()
    handles = [object()]
    add_dll_directory_calls = []

    def add_dll_directory(path):
        add_dll_directory_calls.append(path)
        return handles[0]

    monkeypatch.setattr(libcint.sys, "platform", "win32")
    monkeypatch.setattr(libcint.importlib.resources, "files", lambda package: tmp_path)
    monkeypatch.setattr(libcint.os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.delenv("QCTOOLING_LINKED_LIBARY_PATHS", raising=False)
    libcint._DLL_DIRECTORY_HANDLES.clear()

    libcint._register_linked_libary_paths()

    assert add_dll_directory_calls == [str(bundled_dir)]
    assert libcint._DLL_DIRECTORY_HANDLES == handles


def test_does_not_register_directories_on_non_windows(monkeypatch):
    def add_dll_directory(path):
        raise AssertionError(path)

    monkeypatch.setattr(libcint.sys, "platform", "linux")
    monkeypatch.setattr(libcint.os, "add_dll_directory", add_dll_directory, raising=False)

    libcint._register_linked_libary_paths()
