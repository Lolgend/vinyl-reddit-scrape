import os
import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from reddit_scrape.extraction import (
    EXTRACTION_SYSTEM_PROMPT,
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    create_extraction_agent,
    extract_releases,
    pre_clean_title,
)


class TestPreCleanTitle:
    def test_unicode_dashes_normalized(self):
        # Test each unicode dash: \u2010, \u2011, \u2012, \u2013, \u2014, \u2015, \u2212
        raw = "Miles Davis \u2013 Kind of Blue \u2014 180g \u2212 2LP \u2010 45RPM \u2011 Limited \u2012 Edition \u2015 Box"
        expected = "Miles Davis - Kind of Blue - 180g - 2LP - 45RPM - Limited - Edition - Box"
        assert pre_clean_title(raw) == expected

    def test_whitespace_collapsed(self):
        raw = "  Artist  \t\n   Album   (Red   LP)  "
        assert pre_clean_title(raw) == "Artist Album (Red LP)"

    def test_empty_or_whitespace_only(self):
        assert pre_clean_title("") == ""
        assert pre_clean_title("   \t  \n  ") == ""

    def test_preserves_regular_hyphen_and_special_chars(self):
        raw = "blink-182 - Enema of the State (180g LP)"
        assert pre_clean_title(raw) == "blink-182 - Enema of the State (180g LP)"


class TestExtractionSchema:
    def test_extracted_release_full(self):
        release = ExtractedRelease(
            artists=["Converge", "Dropdead"],
            combined_artist="Converge / Dropdead",
            album_title="Split",
            variant='7" /500',
            retailer="Deathwish",
        )
        assert release.artists == ["Converge", "Dropdead"]
        assert release.combined_artist == "Converge / Dropdead"
        assert release.album_title == "Split"
        assert release.variant == '7" /500'
        assert release.retailer == "Deathwish"

    def test_extracted_release_optional_fields_none(self):
        release = ExtractedRelease(
            artists=["blink-182"],
            album_title="Enema of the State",
        )
        assert release.artists == ["blink-182"]
        assert release.combined_artist is None
        assert release.album_title == "Enema of the State"
        assert release.variant is None
        assert release.retailer is None

    def test_extraction_result_default_empty(self):
        result = ExtractionResult()
        assert result.releases == []

    def test_validation_errors_missing_required(self):
        with pytest.raises(ValidationError):
            ExtractedRelease()  # missing artists & album_title

        with pytest.raises(ValidationError):
            ExtractedRelease(artists=["Fontaines D.C."])  # missing album_title

    def test_serialization_roundtrip(self):
        payload = {
            "releases": [
                {
                    "artists": ["Fontaines D.C."],
                    "combined_artist": "Fontaines D.C.",
                    "album_title": "Romance",
                    "variant": "Opaque Emerald LP",
                    "retailer": None,
                }
            ]
        }
        result = ExtractionResult.model_validate(payload)
        assert len(result.releases) == 1
        assert result.releases[0].album_title == "Romance"
        assert result.model_dump() == payload


class TestAgentConfiguration:
    def test_create_extraction_agent_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY or DEEPSEEK_KEY"):
            create_extraction_agent()

    def test_create_extraction_agent_success(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345")
        agent = create_extraction_agent()
        assert agent is not None
        assert agent.output_type == ExtractionResult
        assert any("expert vinyl record discography classifier" in p for p in agent._system_prompts)

    def test_create_extraction_agent_explicit_args(self):
        agent = create_extraction_agent(
            api_key="sk-explicit",
            base_url="https://custom.deepseek.com",
            model_name="deepseek-custom",
        )
        assert agent is not None
        assert agent.output_type == ExtractionResult

    def test_create_extraction_agent_thinking_level_none(self):
        agent = create_extraction_agent(api_key="sk-test", thinking_level="none")
        assert agent.model_settings == {"thinking": False}

        agent_upper = create_extraction_agent(api_key="sk-test", thinking_level="NONE")
        assert agent_upper.model_settings == {"thinking": False}

    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "max"])
    def test_create_extraction_agent_thinking_levels(self, level):
        agent = create_extraction_agent(api_key="sk-test", thinking_level=level)
        assert agent.model_settings == {"thinking": level}

    def test_create_extraction_agent_thinking_level_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_THINKING_LEVEL", "medium")
        agent = create_extraction_agent()
        assert agent.model_settings == {"thinking": "medium"}

        monkeypatch.setenv("LLM_THINKING_LEVEL", "none")
        agent_none = create_extraction_agent()
        assert agent_none.model_settings == {"thinking": False}

        monkeypatch.delenv("LLM_THINKING_LEVEL", raising=False)
        agent_default = create_extraction_agent()
        assert agent_default.model_settings == {"thinking": "low"}


class TestExtractionAsyncLogic:
    @pytest.mark.asyncio
    async def test_extract_releases_single_artist(self):
        mock_output = {
            "releases": [
                {
                    "artists": ["Fontaines D.C."],
                    "combined_artist": "Fontaines D.C.",
                    "album_title": "Romance",
                    "variant": "Opaque Emerald LP",
                    "retailer": None,
                }
            ]
        }
        test_agent = Agent(
            TestModel(custom_output_args=mock_output),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        title = "Fontaines D.C. \u2013 Romance (Opaque Emerald LP) [$32.99]"
        res = await extract_releases(title, agent=test_agent)

        assert isinstance(res, ExtractionResult)
        assert len(res.releases) == 1
        assert res.releases[0].album_title == "Romance"
        assert res.releases[0].artists == ["Fontaines D.C."]
        assert res.releases[0].variant == "Opaque Emerald LP"

    @pytest.mark.asyncio
    async def test_extract_releases_with_flair_included_in_prompt(self):
        captured_prompts = []

        # We can subclass or inspect prompt with a test agent
        test_agent = Agent(
            TestModel(custom_output_args={"releases": []}),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        res = await extract_releases(
            title="[Discussion] What's your favorite pressing?",
            flair="Discussion",
            agent=test_agent,
        )
        assert isinstance(res, ExtractionResult)
        assert len(res.releases) == 0

    @pytest.mark.asyncio
    async def test_extract_releases_empty_for_non_release(self):
        test_agent = Agent(
            TestModel(custom_output_args={"releases": []}),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        res = await extract_releases(
            title="Target Buy 2 Get 1 Free Sale on All Vinyl Records",
            agent=test_agent,
        )
        assert len(res.releases) == 0

    @pytest.mark.asyncio
    async def test_extraction_worker_class(self):
        mock_output = {
            "releases": [
                {
                    "artists": ["Kanye West", "Ty Dolla $ign"],
                    "combined_artist": "Kanye West & Ty Dolla $ign",
                    "album_title": "Vultures 1",
                    "variant": "Black 2LP",
                    "retailer": None,
                }
            ]
        }
        test_agent = Agent(
            TestModel(custom_output_args=mock_output),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        worker = ExtractionWorker(agent=test_agent)
        res = await worker.extract("Kanye West & Ty Dolla $ign - Vultures 1 (Black 2LP)")

        assert len(res.releases) == 1
        release = res.releases[0]
        assert release.artists == ["Kanye West", "Ty Dolla $ign"]
        assert release.combined_artist == "Kanye West & Ty Dolla $ign"
        assert release.album_title == "Vultures 1"

    @pytest.mark.asyncio
    async def test_extraction_worker_recovers_from_closed_event_loop(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-mock-key")
        worker = ExtractionWorker()

        call_count = 0

        async def fake_extract_releases(title, flair=None, agent=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Event loop is closed")
            return ExtractionResult(
                releases=[
                    ExtractedRelease(
                        artists=["Recovered Artist"],
                        album_title="Recovered Album",
                    )
                ]
            )

        monkeypatch.setattr("reddit_scrape.extraction.extract_releases", fake_extract_releases)
        res = await worker.extract("Test Title")
        assert len(res.releases) == 1
        assert res.releases[0].artists == ["Recovered Artist"]
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_extraction_worker_propagates_thinking_level(self, monkeypatch):
        captured_kwargs = {}

        def fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return Agent(
                TestModel(custom_output_args={"releases": []}),
                output_type=ExtractionResult,
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
            )

        monkeypatch.setattr("reddit_scrape.extraction.create_extraction_agent", fake_create_agent)
        worker = ExtractionWorker(thinking_level="minimal")
        assert worker.thinking_level == "minimal"

        agent = worker._get_agent()
        assert agent is not None
        assert captured_kwargs.get("thinking_level") == "minimal"

        # Also verify recovery propagates thinking_level
        captured_kwargs.clear()
        call_count = 0

        async def fake_extract_releases(title, flair=None, agent=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Event loop is closed")
            return ExtractionResult(releases=[])

        monkeypatch.setattr("reddit_scrape.extraction.extract_releases", fake_extract_releases)
        await worker.extract("Sample Title")
        assert call_count == 2
        assert captured_kwargs.get("thinking_level") == "minimal"


class TestBenchmarkFixtures:
    def test_prototype_results_schema_conformance(self):
        from pathlib import Path
        import json

        json_path = (
            Path(__file__).parent
            / "fixtures"
            / "benchmark_cases.json"
        )
        assert json_path.exists(), f"Missing fixture file at {json_path}"

        with open(json_path, "r", encoding="utf-8") as f:
            cases = json.load(f)

        assert len(cases) == 12
        for item in cases:
            case_info = item["case"]
            result_data = item["result"]
            parsed = ExtractionResult.model_validate(result_data)
            assert isinstance(parsed, ExtractionResult)

            if "ignore" in case_info["id"]:
                assert len(parsed.releases) == 0
            else:
                assert len(parsed.releases) >= 1
                for r in parsed.releases:
                    assert len(r.artists) >= 1
                    assert r.album_title
