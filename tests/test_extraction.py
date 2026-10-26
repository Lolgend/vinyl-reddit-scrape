import json
import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from reddit_scrape.extraction import (
    BATCH_EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    BatchExtractedItem,
    BatchExtractionResult,
    CandidateRetriever,
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    build_batch_extraction_prompt,
    build_extraction_prompt,
    create_batch_extraction_agent,
    create_extraction_agent,
    extract_releases,
    format_catalog_reference,
    get_default_thinking_level,
    pre_clean_title,
)
from reddit_scrape.persistence import CatalogCandidates, CatalogReleaseCandidate


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


class TestBatchExtractionSchema:
    def test_batch_extracted_item_full(self):
        release = ExtractedRelease(
            artists=["Fontaines D.C."],
            combined_artist="Fontaines D.C.",
            album_title="Romance",
            variant="Opaque Emerald LP",
            retailer=None,
        )
        item = BatchExtractedItem(
            submission_id="sub_abc123",
            releases=[release],
        )
        assert item.submission_id == "sub_abc123"
        assert len(item.releases) == 1
        assert item.releases[0].album_title == "Romance"

    def test_batch_extracted_item_default_empty_releases(self):
        item = BatchExtractedItem(submission_id="sub_empty")
        assert item.submission_id == "sub_empty"
        assert item.releases == []

    def test_batch_extracted_item_missing_submission_id(self):
        with pytest.raises(ValidationError):
            BatchExtractedItem()

    def test_batch_extraction_result_default_empty(self):
        result = BatchExtractionResult()
        assert result.items == []

    def test_batch_extraction_result_roundtrip(self):
        payload = {
            "items": [
                {
                    "submission_id": "sub_1",
                    "releases": [
                        {
                            "artists": ["Fontaines D.C."],
                            "combined_artist": "Fontaines D.C.",
                            "album_title": "Romance",
                            "variant": "Green LP",
                            "retailer": "Rough Trade",
                        }
                    ],
                },
                {
                    "submission_id": "sub_2",
                    "releases": [],
                },
            ]
        }
        result = BatchExtractionResult.model_validate(payload)
        assert len(result.items) == 2
        assert result.items[0].submission_id == "sub_1"
        assert result.items[0].releases[0].album_title == "Romance"
        assert result.items[1].submission_id == "sub_2"
        assert result.items[1].releases == []
        assert result.model_dump() == payload


class TestSinglePromptBuilder:
    def test_build_extraction_prompt_with_flair(self):
        prompt = build_extraction_prompt("Fontaines D.C. - Romance", "Official")
        assert "Flair: Official" in prompt
        assert "Title: Fontaines D.C. - Romance" in prompt

    def test_build_extraction_prompt_without_flair(self):
        prompt = build_extraction_prompt("Idles - TANGK", None)
        assert "Flair:" not in prompt
        assert "Idles - TANGK" in prompt


class TestBatchPromptBuilder:
    def test_empty_submissions_returns_empty_string(self):
        assert build_batch_extraction_prompt([]) == ""

    def test_single_submission_with_flair(self):
        prompt = build_batch_extraction_prompt([
            ("sub_101", "Fontaines D.C. - Romance (Opaque Emerald LP)", "Vinyl (Official)")
        ])
        assert "Extract release entities from the following Reddit submissions:" in prompt
        assert "Submission ID: sub_101" in prompt
        assert "Flair: Vinyl (Official)" in prompt
        assert "Title: Fontaines D.C. - Romance (Opaque Emerald LP)" in prompt

    def test_single_submission_without_flair(self):
        prompt = build_batch_extraction_prompt([
            ("sub_102", "Idles - TANGK (Yellow LP)", None)
        ])
        assert "Submission ID: sub_102" in prompt
        assert "Title: Idles - TANGK (Yellow LP)" in prompt
        assert "Flair:" not in prompt

    def test_multiple_submissions_formatting_and_cleaning(self):
        submissions = [
            ("sub_1", "Miles Davis \u2013 Kind of Blue \u2014 180g", "Jazz"),
            ("sub_2", "  blink-182   -   Enema   of the State  ", None),
            ("sub_3", "Discussion: What is your favorite variant?", "   "),
        ]
        prompt = build_batch_extraction_prompt(submissions)
        assert "--- Submission 1 ---" in prompt
        assert "Submission ID: sub_1" in prompt
        assert "Flair: Jazz" in prompt
        assert "Title: Miles Davis - Kind of Blue - 180g" in prompt

        assert "--- Submission 2 ---" in prompt
        assert "Submission ID: sub_2" in prompt
        assert "Title: blink-182 - Enema of the State" in prompt

        assert "--- Submission 3 ---" in prompt
        assert "Submission ID: sub_3" in prompt
        assert "Title: Discussion: What is your favorite variant?" in prompt
        # Whitespace-only flair should not produce Flair: line
        lines = [line.strip() for line in prompt.split("\n")]
        assert "Flair:" not in lines

    def test_build_batch_extraction_prompt_omits_reference_when_none_or_empty(self):
        submissions = [("sub_1", "Fontaines D.C. - Romance", None)]
        assert "Existing Catalog Reference" not in build_batch_extraction_prompt(submissions, candidates=None)
        assert "Existing Catalog Reference" not in build_batch_extraction_prompt(
            submissions, candidates=CatalogCandidates()
        )
        assert "Existing Catalog Reference" not in build_batch_extraction_prompt(
            submissions, candidates=CatalogCandidates(releases=[], artists=[])
        )

    def test_build_batch_extraction_prompt_includes_reference_when_candidates_present(self):
        submissions = [
            ("sub_1", "Romance - Fontaines D.C.", "Vinyl"),
            ("sub_2", "Dropdead / Converge - Split", None),
        ]
        candidates = CatalogCandidates(
            artists=["Fontaines D.C.", "Converge", "Dropdead"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_1",
                    combined_artist="Fontaines D.C.",
                    album_title="Romance",
                    artists=["Fontaines D.C."],
                    variant="Opaque Emerald LP",
                    retailer="XL Recordings",
                ),
                CatalogReleaseCandidate(
                    id="rel_2",
                    combined_artist="Converge / Dropdead",
                    album_title="Split",
                    artists=["Converge", "Dropdead"],
                    variant=None,
                    retailer=None,
                ),
            ],
        )
        prompt = build_batch_extraction_prompt(submissions, candidates=candidates)
        assert "Submission ID: sub_1" in prompt
        assert "Submission ID: sub_2" in prompt
        assert "Existing Catalog Reference:" in prompt
        assert "Canonical Artists:" in prompt
        assert "- Fontaines D.C." in prompt
        assert "- Converge" in prompt
        assert "Canonical Releases:" in prompt
        assert "- Combined Artist: Fontaines D.C. | Artists: Fontaines D.C. | Album Title: Romance | Variant: Opaque Emerald LP | Retailer: XL Recordings" in prompt
        assert "- Combined Artist: Converge / Dropdead | Artists: Converge, Dropdead | Album Title: Split | Variant: None | Retailer: None" in prompt

    def test_build_batch_extraction_prompt_empty_submissions_returns_empty_even_with_candidates(self):
        candidates = CatalogCandidates(artists=["Artist A"])
        assert build_batch_extraction_prompt([], candidates=candidates) == ""


class TestThinkingLevelDefaults:
    def test_get_default_thinking_level_reasoning_models(self):
        assert get_default_thinking_level("deepseek-reasoner") == "low"
        assert get_default_thinking_level("deepseek-r1") == "low"
        assert get_default_thinking_level("DeepSeek-R1-Distill-Qwen-32B") == "low"
        assert get_default_thinking_level("o1-preview") == "low"
        assert get_default_thinking_level("o3-mini") == "low"

    def test_get_default_thinking_level_chat_models(self):
        assert get_default_thinking_level("deepseek-chat") == "none"
        assert get_default_thinking_level("gpt-4o") == "none"
        assert get_default_thinking_level("claude-3-5-sonnet") == "none"

    def test_get_default_thinking_level_env_fallback(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-reasoner")
        assert get_default_thinking_level() == "low"
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")
        assert get_default_thinking_level() == "none"
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        assert get_default_thinking_level() == "none"


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
        # Default for chat model is none -> thinking: False
        agent_default = create_extraction_agent()
        assert agent_default.model_settings == {"thinking": False}

        # Default for reasoning model is low -> thinking: "low"
        agent_reasoner = create_extraction_agent(model_name="deepseek-reasoner")
        assert agent_reasoner.model_settings == {"thinking": "low"}


class TestBatchAgentConfiguration:
    def test_create_batch_extraction_agent_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY or DEEPSEEK_KEY"):
            create_batch_extraction_agent()

    def test_create_batch_extraction_agent_success(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-12345")
        agent = create_batch_extraction_agent()
        assert agent is not None
        assert agent.output_type == BatchExtractionResult
        assert any("batches of r/vinylreleases submission titles" in p for p in agent._system_prompts)

    def test_create_batch_extraction_agent_explicit_args(self):
        agent = create_batch_extraction_agent(
            api_key="sk-explicit",
            base_url="https://custom.deepseek.com",
            model_name="deepseek-custom",
        )
        assert agent is not None
        assert agent.output_type == BatchExtractionResult

    def test_create_batch_extraction_agent_thinking_level_none(self):
        agent = create_batch_extraction_agent(api_key="sk-test", thinking_level="none")
        assert agent.model_settings == {"thinking": False}

        agent_upper = create_batch_extraction_agent(api_key="sk-test", thinking_level="NONE")
        assert agent_upper.model_settings == {"thinking": False}

    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "max"])
    def test_create_batch_extraction_agent_thinking_levels(self, level):
        agent = create_batch_extraction_agent(api_key="sk-test", thinking_level=level)
        assert agent.model_settings == {"thinking": level}

    def test_create_batch_extraction_agent_thinking_level_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_THINKING_LEVEL", "high")
        agent = create_batch_extraction_agent()
        assert agent.model_settings == {"thinking": "high"}

        monkeypatch.setenv("LLM_THINKING_LEVEL", "none")
        agent_none = create_batch_extraction_agent()
        assert agent_none.model_settings == {"thinking": False}

        monkeypatch.delenv("LLM_THINKING_LEVEL", raising=False)
        # Default for chat model is none -> thinking: False
        agent_default = create_batch_extraction_agent()
        assert agent_default.model_settings == {"thinking": False}

        # Default for reasoning model is low -> thinking: "low"
        agent_reasoner = create_batch_extraction_agent(model_name="deepseek-reasoner")
        assert agent_reasoner.model_settings == {"thinking": "low"}


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


class TestBatchExtractionWorker:
    @pytest.mark.asyncio
    async def test_extract_batch_empty_sequence(self):
        worker = ExtractionWorker()
        result = await worker.extract_batch([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_extract_batch_happy_path(self):
        mock_output = {
            "items": [
                {
                    "submission_id": "sub_1",
                    "releases": [
                        {
                            "artists": ["Fontaines D.C."],
                            "combined_artist": "Fontaines D.C.",
                            "album_title": "Romance",
                            "variant": "Opaque Emerald LP",
                            "retailer": None,
                        }
                    ],
                },
                {
                    "submission_id": "sub_2",
                    "releases": [],
                },
            ]
        }
        batch_agent = Agent(
            TestModel(custom_output_args=mock_output),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        worker = ExtractionWorker(batch_agent=batch_agent)
        submissions = [
            ("sub_1", "Fontaines D.C. - Romance (Opaque Emerald LP)", None),
            ("sub_2", "Store restock sale", "Discussion"),
        ]
        results = await worker.extract_batch(submissions)

        assert len(results) == 2
        assert "sub_1" in results and "sub_2" in results
        assert isinstance(results["sub_1"], ExtractionResult)
        assert len(results["sub_1"].releases) == 1
        assert results["sub_1"].releases[0].album_title == "Romance"
        assert results["sub_1"].releases[0].artists == ["Fontaines D.C."]
        assert len(results["sub_2"].releases) == 0

    @pytest.mark.asyncio
    async def test_extract_batch_fallback_on_agent_exception(self):
        # Batch agent will fail when called
        failing_batch_agent = Agent(
            TestModel(),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        async def fail_run(*args, **kwargs):
            raise RuntimeError("Batch LLM call failed unexpectedly")

        failing_batch_agent.run = fail_run

        # Single agent returns individual releases
        single_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Fallback Artist"],
                            "combined_artist": "Fallback Artist",
                            "album_title": "Fallback Album",
                            "variant": None,
                            "retailer": None,
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        worker = ExtractionWorker(agent=single_agent, batch_agent=failing_batch_agent)
        submissions = [
            ("sub_101", "Fallback Artist - Fallback Album", None),
            ("sub_102", "Fallback Artist - Fallback Album", "Official"),
        ]

        results = await worker.extract_batch(submissions)
        assert len(results) == 2
        assert "sub_101" in results and "sub_102" in results
        assert results["sub_101"].releases[0].album_title == "Fallback Album"
        assert results["sub_102"].releases[0].album_title == "Fallback Album"

    @pytest.mark.asyncio
    async def test_extract_batch_fallback_for_omitted_submission_ids(self):
        # Batch output omits sub_2
        mock_partial_output = {
            "items": [
                {
                    "submission_id": "sub_1",
                    "releases": [
                        {
                            "artists": ["Fontaines D.C."],
                            "combined_artist": "Fontaines D.C.",
                            "album_title": "Romance",
                            "variant": "Green LP",
                            "retailer": None,
                        }
                    ],
                }
            ]
        }
        batch_agent = Agent(
            TestModel(custom_output_args=mock_partial_output),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        single_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Idles"],
                            "combined_artist": "Idles",
                            "album_title": "TANGK",
                            "variant": "Yellow LP",
                            "retailer": None,
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )

        worker = ExtractionWorker(agent=single_agent, batch_agent=batch_agent)
        submissions = [
            ("sub_1", "Fontaines D.C. - Romance", None),
            ("sub_2", "Idles - TANGK", None),
        ]

        results = await worker.extract_batch(submissions)
        assert len(results) == 2
        assert "sub_1" in results and "sub_2" in results
        assert results["sub_1"].releases[0].album_title == "Romance"
        assert results["sub_2"].releases[0].album_title == "TANGK"

    @pytest.mark.asyncio
    async def test_extract_batch_closed_event_loop_recovery(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-mock-key")
        worker = ExtractionWorker()

        call_count = 0

        async def fake_run(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Event loop is closed")
            return type(
                "RunResult",
                (),
                {
                    "output": BatchExtractionResult(
                        items=[
                            BatchExtractedItem(
                                submission_id="sub_rec",
                                releases=[
                                    ExtractedRelease(
                                        artists=["Recovered Artist"],
                                        album_title="Recovered Album",
                                    )
                                ],
                            )
                        ]
                    )
                },
            )()

        fake_batch_agent = Agent(
            TestModel(),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )
        fake_batch_agent.run = fake_run

        monkeypatch.setattr(
            "reddit_scrape.extraction.create_batch_extraction_agent",
            lambda **kwargs: fake_batch_agent,
        )

        results = await worker.extract_batch([("sub_rec", "Recovered Artist - Recovered Album", None)])
        assert call_count == 2
        assert len(results) == 1
        assert results["sub_rec"].releases[0].album_title == "Recovered Album"

    @pytest.mark.asyncio
    async def test_extract_batch_propagates_thinking_level(self, monkeypatch):
        captured_kwargs = {}

        def fake_create_batch_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return Agent(
                TestModel(custom_output_args={"items": []}),
                output_type=BatchExtractionResult,
                system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
            )

        monkeypatch.setattr(
            "reddit_scrape.extraction.create_batch_extraction_agent",
            fake_create_batch_agent,
        )
        worker = ExtractionWorker(thinking_level="high")
        assert worker.thinking_level == "high"

        agent = worker._get_batch_agent()
        assert agent is not None
        assert captured_kwargs.get("thinking_level") == "high"

    @pytest.mark.asyncio
    async def test_extract_batch_bounded_batch_size_partitioning(self):
        batch_calls = []

        async def fake_run(prompt):
            sub_count = prompt.count("--- Submission ")
            batch_calls.append(sub_count)
            items = []
            for i in range(1, sub_count + 1):
                pattern = rf"--- Submission {i} ---\s+Submission ID:\s+(\S+)"
                m = re.search(pattern, prompt)
                sub_id = m.group(1) if m else f"sub_{i}"
                items.append(
                    BatchExtractedItem(
                        submission_id=sub_id,
                        releases=[
                            ExtractedRelease(
                                artists=[f"Artist {sub_id}"],
                                album_title=f"Album {sub_id}",
                            )
                        ],
                    )
                )
            return type("RunResult", (), {"output": BatchExtractionResult(items=items)})()

        fake_batch_agent = Agent(
            TestModel(),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )
        fake_batch_agent.run = fake_run

        worker = ExtractionWorker(batch_agent=fake_batch_agent)
        submissions = [
            (f"sub_{i}", f"Artist {i} - Album {i}", None)
            for i in range(7)
        ]

        results = await worker.extract_batch(submissions, batch_size=3)
        assert len(results) == 7
        assert batch_calls == [3, 3, 1]
        for i in range(7):
            assert f"sub_{i}" in results
            assert results[f"sub_{i}"].releases[0].album_title == f"Album sub_{i}"

    @pytest.mark.asyncio
    async def test_extract_batch_omission_fallback_exception_handling(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)

        # Batch agent returns item for sub_1 only, omitting sub_2 and sub_3
        mock_output = {
            "items": [
                {
                    "submission_id": "sub_1",
                    "releases": [
                        {
                            "artists": ["Artist 1"],
                            "album_title": "Album 1",
                        }
                    ],
                }
            ]
        }
        batch_agent = Agent(
            TestModel(custom_output_args=mock_output),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        worker = ExtractionWorker(batch_agent=batch_agent)

        # Single extract will succeed for sub_2, but raise Exception for sub_3
        async def mock_extract(title, flair=None):
            if "Album 3" in title:
                raise RuntimeError("LLM rate limit error on single item")
            return ExtractionResult(
                releases=[
                    ExtractedRelease(
                        artists=["Artist 2"],
                        album_title="Album 2",
                    )
                ]
            )

        worker.extract = mock_extract

        submissions = [
            ("sub_1", "Artist 1 - Album 1", None),
            ("sub_2", "Artist 2 - Album 2", None),
            ("sub_3", "Artist 3 - Album 3", None),
        ]

        results = await worker.extract_batch(submissions)
        assert len(results) == 3
        assert len(results["sub_1"].releases) == 1
        assert results["sub_1"].releases[0].album_title == "Album 1"
        assert len(results["sub_2"].releases) == 1
        assert results["sub_2"].releases[0].album_title == "Album 2"
        # sub_3 failed during fallback extraction, should have empty releases without raising
        assert len(results["sub_3"].releases) == 0
        assert any(
            "Fallback extraction failed for omitted submission sub_3" in record.message
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_extract_batch_candidate_injection_and_single_retriever_call_per_chunk(self):
        retriever_calls: list[Sequence[str]] = []
        captured_prompts: list[str] = []

        canonical_candidates = CatalogCandidates(
            artists=["Converge", "Dropdead"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_split",
                    combined_artist="Converge / Dropdead",
                    album_title="Split",
                    artists=["Converge", "Dropdead"],
                    variant='7" /500',
                    retailer="Deathwish",
                )
            ],
        )

        def mock_retriever(titles: Sequence[str]) -> CatalogCandidates:
            retriever_calls.append(list(titles))
            return canonical_candidates

        batch_agent = Agent(
            TestModel(
                custom_output_args={
                    "items": [
                        {
                            "submission_id": "sub_1",
                            "releases": [
                                {
                                    "artists": ["Converge", "Dropdead"],
                                    "combined_artist": "Converge / Dropdead",
                                    "album_title": "Split",
                                    "variant": '7" /500',
                                    "retailer": "Deathwish",
                                }
                            ],
                        },
                        {
                            "submission_id": "sub_2",
                            "releases": [
                                {
                                    "artists": ["Converge", "Dropdead"],
                                    "combined_artist": "Converge / Dropdead",
                                    "album_title": "Split",
                                    "variant": '7" /500',
                                    "retailer": "Deathwish",
                                }
                            ],
                        },
                    ]
                }
            ),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        original_run = batch_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        batch_agent.run = spy_run

        worker = ExtractionWorker(batch_agent=batch_agent, candidate_retriever=mock_retriever)
        submissions = [
            ("sub_1", "Dropdead / Converge - Split 7\" /500", "Official"),
            ("sub_2", "Converge / Dropdead - Split (Deathwish)", None),
        ]

        results = await worker.extract_batch(submissions)

        assert len(results) == 2
        # Retriever called exactly once with all titles in the chunk
        assert len(retriever_calls) == 1
        assert retriever_calls[0] == [
            "Dropdead / Converge - Split 7\" /500",
            "Converge / Dropdead - Split (Deathwish)",
        ]

        # Prompt received by batch agent includes Existing Catalog Reference
        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Canonical Artists:" in captured_prompts[0]
        assert "- Converge" in captured_prompts[0]
        assert "Canonical Releases:" in captured_prompts[0]
        assert "Converge / Dropdead" in captured_prompts[0]
        assert "Split" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_batch_chunking_retrieves_candidates_per_chunk(self):
        retriever_calls: list[Sequence[str]] = []
        captured_prompts: list[str] = []

        def mock_retriever(titles: Sequence[str]) -> CatalogCandidates:
            retriever_calls.append(list(titles))
            return CatalogCandidates(artists=[f"Artist for {len(titles)} titles"])

        async def fake_run(prompt):
            captured_prompts.append(prompt)
            sub_count = prompt.count("--- Submission ")
            items = []
            for i in range(1, sub_count + 1):
                pattern = rf"--- Submission {i} ---\s+Submission ID:\s+(\S+)"
                m = re.search(pattern, prompt)
                sub_id = m.group(1) if m else f"sub_{i}"
                items.append(
                    BatchExtractedItem(
                        submission_id=sub_id,
                        releases=[
                            ExtractedRelease(
                                artists=[f"Artist {sub_id}"],
                                album_title=f"Album {sub_id}",
                            )
                        ],
                    )
                )
            return type("RunResult", (), {"output": BatchExtractionResult(items=items)})()

        fake_batch_agent = Agent(
            TestModel(),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )
        fake_batch_agent.run = fake_run

        worker = ExtractionWorker(batch_agent=fake_batch_agent, candidate_retriever=mock_retriever)
        submissions = [
            ("sub_1", "Artist 1 - Album 1", None),
            ("sub_2", "Artist 2 - Album 2", None),
            ("sub_3", "Artist 3 - Album 3", None),
            ("sub_4", "Artist 4 - Album 4", None),
        ]

        results = await worker.extract_batch(submissions, batch_size=2)
        assert len(results) == 4
        # Retriever called once per chunk of 2
        assert len(retriever_calls) == 2
        assert retriever_calls[0] == ["Artist 1 - Album 1", "Artist 2 - Album 2"]
        assert retriever_calls[1] == ["Artist 3 - Album 3", "Artist 4 - Album 4"]
        assert len(captured_prompts) == 2
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Existing Catalog Reference:" in captured_prompts[1]

    @pytest.mark.asyncio
    async def test_extract_batch_preserves_candidates_during_batch_failure_fallback(self):
        failing_batch_agent = Agent(
            TestModel(),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        async def fail_run(*args, **kwargs):
            raise RuntimeError("Batch LLM call crashed")

        failing_batch_agent.run = fail_run

        single_prompts: list[str] = []
        single_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Fallback Artist"],
                            "combined_artist": "Fallback Artist",
                            "album_title": "Fallback Album",
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_single_run = single_agent.run

        async def spy_single_run(prompt, *args, **kwargs):
            single_prompts.append(prompt)
            return await original_single_run(prompt, *args, **kwargs)

        single_agent.run = spy_single_run

        canonical_candidates = CatalogCandidates(
            artists=["Fallback Artist"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_fb",
                    combined_artist="Fallback Artist",
                    album_title="Fallback Album",
                    artists=["Fallback Artist"],
                )
            ],
        )

        worker = ExtractionWorker(
            agent=single_agent,
            batch_agent=failing_batch_agent,
            candidate_retriever=lambda _: canonical_candidates,
        )

        submissions = [
            ("sub_101", "Fallback Album - Fallback Artist", None),
            ("sub_102", "Fallback Artist - Fallback Album", "News"),
        ]

        results = await worker.extract_batch(submissions)
        assert len(results) == 2
        assert "sub_101" in results and "sub_102" in results
        # Fallback single agent prompts must include Existing Catalog Reference
        assert len(single_prompts) == 2
        for prompt in single_prompts:
            assert "Existing Catalog Reference:" in prompt
            assert "Fallback Artist" in prompt
            assert "Fallback Album" in prompt

    @pytest.mark.asyncio
    async def test_extract_batch_preserves_candidates_during_omitted_id_fallback(self):
        # Batch agent returns only sub_1, omitting sub_2
        mock_output = {
            "items": [
                {
                    "submission_id": "sub_1",
                    "releases": [
                        {
                            "artists": ["Fontaines D.C."],
                            "combined_artist": "Fontaines D.C.",
                            "album_title": "Romance",
                        }
                    ],
                }
            ]
        }
        batch_agent = Agent(
            TestModel(custom_output_args=mock_output),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )

        single_prompts: list[str] = []
        single_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Idles"],
                            "combined_artist": "Idles",
                            "album_title": "TANGK",
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_single_run = single_agent.run

        async def spy_single_run(prompt, *args, **kwargs):
            single_prompts.append(prompt)
            return await original_single_run(prompt, *args, **kwargs)

        single_agent.run = spy_single_run

        canonical_candidates = CatalogCandidates(
            artists=["Idles"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_idles",
                    combined_artist="Idles",
                    album_title="TANGK",
                    artists=["Idles"],
                )
            ],
        )

        worker = ExtractionWorker(
            agent=single_agent,
            batch_agent=batch_agent,
            candidate_retriever=lambda _: canonical_candidates,
        )

        submissions = [
            ("sub_1", "Fontaines D.C. - Romance", None),
            ("sub_2", "TANGK - Idles", None),
        ]

        results = await worker.extract_batch(submissions)
        assert len(results) == 2
        assert "sub_1" in results and "sub_2" in results
        # Single prompt for sub_2 must contain Existing Catalog Reference
        assert len(single_prompts) == 1
        assert "Existing Catalog Reference:" in single_prompts[0]
        assert "TANGK" in single_prompts[0]
        assert "Idles" in single_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_batch_explicit_candidates_and_retriever_override(self):
        worker_retriever_called = False

        def worker_retriever(titles):
            nonlocal worker_retriever_called
            worker_retriever_called = True
            return CatalogCandidates(artists=["Worker Artist"])

        explicit_candidates = CatalogCandidates(
            artists=["Explicit Artist"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_exp",
                    combined_artist="Explicit Artist",
                    album_title="Explicit Album",
                    artists=["Explicit Artist"],
                )
            ],
        )

        captured_prompts: list[str] = []
        batch_agent = Agent(
            TestModel(
                custom_output_args={
                    "items": [
                        {
                            "submission_id": "sub_exp",
                            "releases": [
                                {
                                    "artists": ["Explicit Artist"],
                                    "album_title": "Explicit Album",
                                }
                            ],
                        }
                    ]
                }
            ),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = batch_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        batch_agent.run = spy_run

        worker = ExtractionWorker(batch_agent=batch_agent, candidate_retriever=worker_retriever)

        results = await worker.extract_batch(
            [("sub_exp", "Explicit Album - Explicit Artist", None)],
            candidates=explicit_candidates,
        )

        assert len(results) == 1
        # Worker retriever should NOT be called when explicit candidates are passed
        assert not worker_retriever_called
        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Explicit Artist" in captured_prompts[0]
        assert "Explicit Album" in captured_prompts[0]


class TestBenchmarkFixtures:
    def test_prototype_results_schema_conformance(self):
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


class TestCatalogReferenceFormatting:
    def test_format_catalog_reference_none_or_empty(self):
        assert format_catalog_reference(None) == ""
        assert format_catalog_reference(CatalogCandidates()) == ""
        assert format_catalog_reference(CatalogCandidates(releases=[], artists=[])) == ""

    def test_format_catalog_reference_artists_only(self):
        candidates = CatalogCandidates(artists=["Converge", "Dropdead"])
        ref = format_catalog_reference(candidates)
        assert "Existing Catalog Reference:" in ref
        assert "Canonical Artists:" in ref
        assert "- Converge" in ref
        assert "- Dropdead" in ref
        assert "Canonical Releases:" not in ref

    def test_format_catalog_reference_releases_only(self):
        candidates = CatalogCandidates(
            releases=[
                CatalogReleaseCandidate(
                    id="rel_1",
                    combined_artist="Converge / Dropdead",
                    album_title="Split",
                    artists=["Converge", "Dropdead"],
                    variant='7" /500',
                    retailer="Deathwish",
                )
            ]
        )
        ref = format_catalog_reference(candidates)
        assert "Existing Catalog Reference:" in ref
        assert "Canonical Releases:" in ref
        assert "Canonical Artists:" not in ref
        assert "- Combined Artist: Converge / Dropdead | Artists: Converge, Dropdead | Album Title: Split | Variant: 7\" /500 | Retailer: Deathwish" in ref

    def test_format_catalog_reference_both_artists_and_releases(self):
        candidates = CatalogCandidates(
            artists=["Converge", "Dropdead"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_1",
                    combined_artist="Converge / Dropdead",
                    album_title="Split",
                    artists=["Converge", "Dropdead"],
                    variant=None,
                    retailer=None,
                )
            ],
        )
        ref = format_catalog_reference(candidates)
        assert "Existing Catalog Reference:" in ref
        assert "Canonical Artists:" in ref
        assert "- Converge" in ref
        assert "Canonical Releases:" in ref
        assert "- Combined Artist: Converge / Dropdead | Artists: Converge, Dropdead | Album Title: Split | Variant: None | Retailer: None" in ref

    def test_build_extraction_prompt_omits_reference_when_no_candidates(self):
        prompt = build_extraction_prompt("Fontaines D.C. - Romance", candidates=None)
        assert "Existing Catalog Reference" not in prompt
        assert prompt == "Extract release entities from Reddit submission title:\n\nFontaines D.C. - Romance"

        empty_prompt = build_extraction_prompt(
            "Fontaines D.C. - Romance",
            candidates=CatalogCandidates(),
        )
        assert "Existing Catalog Reference" not in empty_prompt

    def test_build_extraction_prompt_includes_reference_when_candidates_present(self):
        candidates = CatalogCandidates(
            artists=["Fontaines D.C."],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_1",
                    combined_artist="Fontaines D.C.",
                    album_title="Romance",
                    artists=["Fontaines D.C."],
                    variant="Opaque Emerald LP",
                    retailer=None,
                )
            ],
        )
        prompt = build_extraction_prompt(
            "Romance - Fontaines D.C.",
            flair="Vinyl (Official)",
            candidates=candidates,
        )
        assert "Flair: Vinyl (Official)" in prompt
        assert "Title: Romance - Fontaines D.C." in prompt
        assert "Existing Catalog Reference:" in prompt
        assert "Canonical Artists:\n- Fontaines D.C." in prompt
        assert "- Combined Artist: Fontaines D.C. | Artists: Fontaines D.C. | Album Title: Romance | Variant: Opaque Emerald LP | Retailer: None" in prompt


class TestSystemPromptCanonicalAlignment:
    def test_system_prompt_includes_canonical_alignment_section(self):
        assert "Canonical Alignment" in EXTRACTION_SYSTEM_PROMPT
        assert "Existing Catalog Reference" in EXTRACTION_SYSTEM_PROMPT

    def test_system_prompt_covers_reordered_and_billing_variations(self):
        assert "Exact & Reordered Match Alignment" in EXTRACTION_SYSTEM_PROMPT
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "revers" in prompt_lower or "order" in prompt_lower
        assert "combined_artist" in EXTRACTION_SYSTEM_PROMPT
        assert "album_title" in EXTRACTION_SYSTEM_PROMPT

    def test_system_prompt_covers_partial_match_new_variant(self):
        assert "Partial Match Support" in EXTRACTION_SYSTEM_PROMPT
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "new variant" in prompt_lower or "variant" in prompt_lower

    def test_system_prompt_covers_anti_hallucination_preservation(self):
        assert "Anti-Hallucination" in EXTRACTION_SYSTEM_PROMPT
        prompt_lower = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "different" in prompt_lower or "unrelated" in prompt_lower


class TestBatchSystemPromptCanonicalAlignment:
    def test_batch_system_prompt_includes_catalog_canonical_alignment(self):
        assert "Canonical Alignment" in BATCH_EXTRACTION_SYSTEM_PROMPT
        assert "Existing Catalog Reference" in BATCH_EXTRACTION_SYSTEM_PROMPT
        assert "Exact & Reordered Match Alignment" in BATCH_EXTRACTION_SYSTEM_PROMPT
        assert "Partial Match Support" in BATCH_EXTRACTION_SYSTEM_PROMPT
        assert "Anti-Hallucination" in BATCH_EXTRACTION_SYSTEM_PROMPT

    def test_batch_system_prompt_includes_intra_batch_canonical_alignment(self):
        prompt_lower = BATCH_EXTRACTION_SYSTEM_PROMPT.lower()
        assert "intra-batch" in prompt_lower or "intra batch" in prompt_lower
        assert "canonical" in prompt_lower
        assert "order" in prompt_lower or "consistent" in prompt_lower
        assert "batch" in prompt_lower


class TestCandidateRetrieverWiring:
    @pytest.mark.asyncio
    async def test_extract_releases_standalone_without_retriever(self):
        mock_output = {"releases": [{"artists": ["Artist A"], "album_title": "Album A"}]}
        test_agent = Agent(
            TestModel(custom_output_args=mock_output),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        captured_prompts = []
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        result = await extract_releases("Artist A - Album A", agent=test_agent)
        assert len(result.releases) == 1
        assert "Existing Catalog Reference" not in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_worker_sync_retriever_wires_candidates(self):
        captured_prompts = []
        test_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Converge", "Dropdead"],
                            "combined_artist": "Converge / Dropdead",
                            "album_title": "Split",
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        retriever_called_with = []

        mock_retriever: CandidateRetriever

        def mock_retriever(titles: Sequence[str]) -> CatalogCandidates:
            retriever_called_with.append(list(titles))
            return CatalogCandidates(
                artists=["Converge", "Dropdead"],
                releases=[
                    CatalogReleaseCandidate(
                        id="rel_split",
                        combined_artist="Converge / Dropdead",
                        album_title="Split",
                        artists=["Converge", "Dropdead"],
                        variant='7" /500',
                        retailer="Deathwish",
                    )
                ],
            )

        worker = ExtractionWorker(agent=test_agent, candidate_retriever=mock_retriever)
        res = await worker.extract("Dropdead / Converge - Split")

        assert len(res.releases) == 1
        assert retriever_called_with == [["Dropdead / Converge - Split"]]
        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Canonical Artists:\n- Converge\n- Dropdead" in captured_prompts[0]
        assert "Combined Artist: Converge / Dropdead" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_worker_async_retriever_wires_candidates(self):
        captured_prompts = []
        test_agent = Agent(
            TestModel(custom_output_args={"releases": []}),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        async def async_retriever(titles: Sequence[str]) -> CatalogCandidates:
            return CatalogCandidates(artists=["Miles Davis"])

        worker = ExtractionWorker(agent=test_agent, candidate_retriever=async_retriever)
        await worker.extract("Miles Davis - Kind of Blue")

        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Canonical Artists:\n- Miles Davis" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_worker_override_retriever(self):
        captured_prompts = []
        test_agent = Agent(
            TestModel(custom_output_args={"releases": []}),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        def default_retriever(titles: Sequence[str]) -> CatalogCandidates:
            return CatalogCandidates(artists=["Default Artist"])

        def override_retriever(titles: Sequence[str]) -> CatalogCandidates:
            return CatalogCandidates(artists=["Override Artist"])

        worker = ExtractionWorker(agent=test_agent, candidate_retriever=default_retriever)
        await worker.extract("Test Title", candidate_retriever=override_retriever)

        assert len(captured_prompts) == 1
        assert "Override Artist" in captured_prompts[0]
        assert "Default Artist" not in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_worker_explicit_candidates_bypasses_retriever(self):
        captured_prompts = []
        test_agent = Agent(
            TestModel(custom_output_args={"releases": []}),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        retriever_called = False

        def mock_retriever(titles: Sequence[str]) -> CatalogCandidates:
            nonlocal retriever_called
            retriever_called = True
            return CatalogCandidates(artists=["Retrieved Artist"])

        explicit = CatalogCandidates(artists=["Explicit Artist"])
        worker = ExtractionWorker(agent=test_agent, candidate_retriever=mock_retriever)
        await worker.extract("Test Title", candidates=explicit)

        assert not retriever_called
        assert len(captured_prompts) == 1
        assert "Explicit Artist" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_extract_worker_retriever_failure_graceful_degradation(self, caplog):
        captured_prompts = []
        test_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Safe Artist"],
                            "album_title": "Safe Album",
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        def failing_retriever(titles: Sequence[str]) -> CatalogCandidates:
            raise ConnectionError("Database connection lost")

        worker = ExtractionWorker(agent=test_agent, candidate_retriever=failing_retriever)
        result = await worker.extract("Safe Artist - Safe Album")

        assert len(result.releases) == 1
        assert result.releases[0].album_title == "Safe Album"
        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference" not in captured_prompts[0]
        assert any("Candidate retrieval failed" in record.message for record in caplog.records)


class TestCanonicalAlignmentBehavior:
    @pytest.mark.asyncio
    async def test_reordered_artist_album_prompt_and_alignment(self):
        canonical_candidates = CatalogCandidates(
            artists=["Miles Davis"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_kob",
                    combined_artist="Miles Davis",
                    album_title="Kind of Blue",
                    artists=["Miles Davis"],
                    variant="180g Mono LP",
                    retailer="Sony Legacy",
                )
            ],
        )

        test_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Miles Davis"],
                            "combined_artist": "Miles Davis",
                            "album_title": "Kind of Blue",
                            "variant": "180g Mono LP",
                            "retailer": "Sony Legacy",
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        captured_prompts = []
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        worker = ExtractionWorker(
            agent=test_agent,
            candidate_retriever=lambda _: canonical_candidates,
        )
        res = await worker.extract("Kind of Blue - Miles Davis (180g Mono LP) [Sony Legacy]")

        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Kind of Blue" in captured_prompts[0]
        assert "Miles Davis" in captured_prompts[0]

        assert len(res.releases) == 1
        rel = res.releases[0]
        assert rel.artists == ["Miles Davis"]
        assert rel.combined_artist == "Miles Davis"
        assert rel.album_title == "Kind of Blue"

    @pytest.mark.asyncio
    async def test_partial_match_new_variant_prompt_and_alignment(self):
        canonical_candidates = CatalogCandidates(
            artists=["Fontaines D.C."],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_romance",
                    combined_artist="Fontaines D.C.",
                    album_title="Romance",
                    artists=["Fontaines D.C."],
                    variant="Opaque Emerald LP",
                    retailer="XL Recordings",
                )
            ],
        )

        test_agent = Agent(
            TestModel(
                custom_output_args={
                    "releases": [
                        {
                            "artists": ["Fontaines D.C."],
                            "combined_artist": "Fontaines D.C.",
                            "album_title": "Romance",
                            "variant": "Clear Zoetrope LP",
                            "retailer": "Blood Records",
                        }
                    ]
                }
            ),
            output_type=ExtractionResult,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
        )
        captured_prompts = []
        original_run = test_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        test_agent.run = spy_run

        worker = ExtractionWorker(
            agent=test_agent,
            candidate_retriever=lambda _: canonical_candidates,
        )
        res = await worker.extract("Romance - Fontaines D.C. (Clear Zoetrope LP) [Blood Records]")

        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]

        assert len(res.releases) == 1
        rel = res.releases[0]
        assert rel.artists == ["Fontaines D.C."]
        assert rel.combined_artist == "Fontaines D.C."
        assert rel.album_title == "Romance"
        assert rel.variant == "Clear Zoetrope LP"
        assert rel.retailer == "Blood Records"

    @pytest.mark.asyncio
    async def test_batch_extraction_intra_batch_canonical_alignment_with_catalog_reference(self):
        canonical_candidates = CatalogCandidates(
            artists=["Converge", "Dropdead"],
            releases=[
                CatalogReleaseCandidate(
                    id="rel_split",
                    combined_artist="Converge / Dropdead",
                    album_title="Split",
                    artists=["Converge", "Dropdead"],
                    variant='7"',
                    retailer=None,
                )
            ],
        )

        captured_prompts: list[str] = []
        batch_agent = Agent(
            TestModel(
                custom_output_args={
                    "items": [
                        {
                            "submission_id": "sub_1",
                            "releases": [
                                {
                                    "artists": ["Converge", "Dropdead"],
                                    "combined_artist": "Converge / Dropdead",
                                    "album_title": "Split",
                                    "variant": '7"',
                                    "retailer": None,
                                }
                            ],
                        },
                        {
                            "submission_id": "sub_2",
                            "releases": [
                                {
                                    "artists": ["Converge", "Dropdead"],
                                    "combined_artist": "Converge / Dropdead",
                                    "album_title": "Split",
                                    "variant": '7"',
                                    "retailer": None,
                                }
                            ],
                        },
                    ]
                }
            ),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = batch_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        batch_agent.run = spy_run

        worker = ExtractionWorker(
            batch_agent=batch_agent,
            candidate_retriever=lambda _: canonical_candidates,
        )

        submissions = [
            ("sub_1", "Dropdead / Converge - Split 7\"", None),
            ("sub_2", "Split 7\" - Converge / Dropdead", None),
        ]

        results = await worker.extract_batch(submissions)

        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" in captured_prompts[0]
        assert "Converge / Dropdead" in captured_prompts[0]

        assert len(results) == 2
        rel1 = results["sub_1"].releases[0]
        rel2 = results["sub_2"].releases[0]
        assert rel1.combined_artist == "Converge / Dropdead"
        assert rel2.combined_artist == "Converge / Dropdead"
        assert rel1.album_title == "Split"
        assert rel2.album_title == "Split"
        assert rel1.artists == rel2.artists == ["Converge", "Dropdead"]

    @pytest.mark.asyncio
    async def test_batch_extraction_intra_batch_canonical_alignment_without_catalog_reference(self):
        captured_prompts: list[str] = []
        batch_agent = Agent(
            TestModel(
                custom_output_args={
                    "items": [
                        {
                            "submission_id": "sub_1",
                            "releases": [
                                {
                                    "artists": ["Converge", "Dropdead"],
                                    "combined_artist": "Converge / Dropdead",
                                    "album_title": "Unreleased",
                                    "variant": '12"',
                                    "retailer": None,
                                }
                            ],
                        },
                        {
                            "submission_id": "sub_2",
                            "releases": [
                                {
                                    "artists": ["Converge", "Dropdead"],
                                    "combined_artist": "Converge / Dropdead",
                                    "album_title": "Unreleased",
                                    "variant": '12"',
                                    "retailer": None,
                                }
                            ],
                        },
                    ]
                }
            ),
            output_type=BatchExtractionResult,
            system_prompt=BATCH_EXTRACTION_SYSTEM_PROMPT,
        )
        original_run = batch_agent.run

        async def spy_run(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return await original_run(prompt, *args, **kwargs)

        batch_agent.run = spy_run

        worker = ExtractionWorker(
            batch_agent=batch_agent,
            candidate_retriever=lambda _: CatalogCandidates(),
        )

        submissions = [
            ("sub_1", "Dropdead / Converge - Unreleased 12\"", None),
            ("sub_2", "Unreleased 12\" - Converge / Dropdead", None),
        ]

        results = await worker.extract_batch(submissions)

        assert len(captured_prompts) == 1
        assert "Existing Catalog Reference:" not in captured_prompts[0]

        assert len(results) == 2
        rel1 = results["sub_1"].releases[0]
        rel2 = results["sub_2"].releases[0]
        assert rel1.combined_artist == rel2.combined_artist == "Converge / Dropdead"
        assert rel1.album_title == rel2.album_title == "Unreleased"
        assert rel1.artists == rel2.artists == ["Converge", "Dropdead"]
