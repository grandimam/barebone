import pytest

from barebone.tools import _build_pydantic_model
from barebone.tools import _extract_description
from barebone.tools import _parse_response
from barebone.tools import edit
from barebone.tools import glob
from barebone.tools import grep
from barebone.tools import Question
from barebone.tools import QuestionOption
from barebone.tools import read
from barebone.tools import tool
from barebone.tools import write


class TestToolDecorator:
    def test_basic_decorator(self):
        @tool
        def my_func(x: int) -> int:
            return x * 2

        assert my_func.name == "my_func"
        assert my_func(5) == 10

    def test_decorator_with_name(self):
        @tool("custom_name")
        def my_func(x: int) -> int:
            return x

        assert my_func.name == "custom_name"

    def test_decorator_with_description(self):
        @tool(description="Custom description")
        def my_func(x: int) -> int:
            return x

        tool_def = my_func.to_tool()
        assert tool_def.description == "Custom description"

    def test_decorator_with_name_and_description(self):
        @tool("custom_name", description="Custom description")
        def my_func(x: int) -> int:
            return x

        assert my_func.name == "custom_name"
        tool_def = my_func.to_tool()
        assert tool_def.description == "Custom description"


class TestExtractDescription:
    def test_with_docstring(self):
        def my_func():
            """This is the description."""
            pass

        assert _extract_description(my_func) == "This is the description."

    def test_with_multiline_docstring(self):
        def my_func():
            """First paragraph here.

            Second paragraph.
            """
            pass

        assert _extract_description(my_func) == "First paragraph here."

    def test_without_docstring(self):
        def my_func():
            pass

        assert _extract_description(my_func) == "Execute my_func"


class TestBuildPydanticModel:
    def test_simple_params(self):
        def my_func(name: str, count: int) -> str:
            return f"{name}: {count}"

        model = _build_pydantic_model(my_func)
        instance = model(name="test", count=5)
        assert instance.name == "test"
        assert instance.count == 5

    def test_with_defaults(self):
        def my_func(name: str, count: int = 10) -> str:
            return f"{name}: {count}"

        model = _build_pydantic_model(my_func)
        instance = model(name="test")
        assert instance.count == 10

    def test_skips_self(self):
        def my_method(self, name: str) -> str:
            return name

        model = _build_pydantic_model(my_method)
        schema = model.model_json_schema()
        assert "self" not in schema.get("properties", {})


class TestToolWrapper:
    def test_to_tool(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        tool_def = add.to_tool()
        assert tool_def.name == "add"
        assert tool_def.description == "Add two numbers."
        assert "a" in tool_def.parameters["properties"]
        assert "b" in tool_def.parameters["properties"]
        assert tool_def.handler(1, 2) == 3

    def test_callable(self):
        @tool
        def multiply(x: int, y: int) -> int:
            return x * y

        assert multiply(3, 4) == 12


class TestParseResponse:
    def test_single_selection(self):
        options = [
            QuestionOption(label="Option A", description="First option"),
            QuestionOption(label="Option B", description="Second option"),
        ]
        assert _parse_response("1", options, False) == "Option A"
        assert _parse_response("2", options, False) == "Option B"

    def test_multi_selection(self):
        options = [
            QuestionOption(label="A", description=""),
            QuestionOption(label="B", description=""),
            QuestionOption(label="C", description=""),
        ]
        assert _parse_response("1, 3", options, True) == "A, C"

    def test_custom_input(self):
        options = [QuestionOption(label="A", description="")]
        assert _parse_response("custom answer", options, False) == "custom answer"

    def test_empty_input(self):
        options = [QuestionOption(label="A", description="")]
        assert _parse_response("", options, False) == "(no response)"


class TestQuestionModels:
    def test_question_option(self):
        opt = QuestionOption(label="Yes", description="Confirm action")
        assert opt.label == "Yes"
        assert opt.description == "Confirm action"

    def test_question(self):
        q = Question(
            question="Continue?",
            header="Confirm",
            options=[QuestionOption(label="Yes", description="")],
            multiSelect=False,
        )
        assert q.question == "Continue?"
        assert q.header == "Confirm"
        assert len(q.options) == 1


class TestReadTool:
    def test_read_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\n")

        result = read(str(test_file))
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_read_with_offset(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        result = read(str(test_file), offset=2, limit=2)
        assert "line1" not in result
        assert "line2" not in result
        assert "line3" in result
        assert "line4" in result
        assert "line5" not in result

    def test_read_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read(str(tmp_path / "nonexistent.txt"))

    def test_read_directory(self, tmp_path):
        with pytest.raises(IsADirectoryError):
            read(str(tmp_path))


class TestWriteTool:
    def test_write_new_file(self, tmp_path):
        test_file = tmp_path / "new_file.txt"
        result = write(str(test_file), "Hello, World!")

        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"
        assert "Successfully wrote" in result

    def test_write_creates_directories(self, tmp_path):
        test_file = tmp_path / "subdir" / "nested" / "file.txt"
        write(str(test_file), "nested content")

        assert test_file.exists()
        assert test_file.read_text() == "nested content"

    def test_write_overwrites(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("original")

        write(str(test_file), "updated")
        assert test_file.read_text() == "updated"


class TestEditTool:
    def test_edit_replaces_string(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        result = edit(str(test_file), "World", "Universe")
        assert test_file.read_text() == "Hello Universe"
        assert "Successfully edited" in result

    def test_edit_nonexistent_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            edit(str(tmp_path / "nonexistent.txt"), "old", "new")

    def test_edit_string_not_found(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        with pytest.raises(ValueError, match="String not found"):
            edit(str(test_file), "nonexistent string", "new")

    def test_edit_ambiguous_string(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("foo bar foo baz")

        with pytest.raises(ValueError, match="appears 2 times"):
            edit(str(test_file), "foo", "qux")


class TestGlobTool:
    def test_glob_finds_files(self, tmp_path):
        (tmp_path / "file1.py").touch()
        (tmp_path / "file2.py").touch()
        (tmp_path / "file.txt").touch()

        result = glob("*.py", str(tmp_path))
        assert "file1.py" in result
        assert "file2.py" in result
        assert "file.txt" not in result

    def test_glob_recursive(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").touch()
        (tmp_path / "root.py").touch()

        result = glob("**/*.py", str(tmp_path))
        assert "nested.py" in result
        assert "root.py" in result

    def test_glob_no_matches(self, tmp_path):
        result = glob("*.xyz", str(tmp_path))
        assert "No files matching" in result

    def test_glob_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError):
            glob("*.py", "/nonexistent/path")


class TestGrepTool:
    def test_grep_finds_pattern(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('world')\n")

        result = grep("hello", str(tmp_path))
        assert "hello" in result
        assert str(test_file) in result

    def test_grep_regex_pattern(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("def func1():\ndef func2():\n")

        result = grep(r"def func\d", str(tmp_path))
        assert "func1" in result
        assert "func2" in result

    def test_grep_no_matches(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = grep("nonexistent_pattern", str(tmp_path))
        assert "No matches" in result

    def test_grep_with_file_glob(self, tmp_path):
        (tmp_path / "test.py").write_text("python code")
        (tmp_path / "test.txt").write_text("python text")

        result = grep("python", str(tmp_path), file_glob="*.py")
        assert "test.py" in result
        assert "test.txt" not in result
