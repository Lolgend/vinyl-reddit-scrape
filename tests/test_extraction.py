import json
import re
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
    ExtractedRelease,
    ExtractionResult,
    ExtractionWorker,
    build_batch_extraction_prompt,
    build_extraction_prompt,
    create_batch_extraction_agent,
    create_extraction_agent,
    extract_releases,
    get_default_thinking_level,
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
