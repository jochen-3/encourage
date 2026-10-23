import unittest

from encourage.llm import Response, ResponseWrapper
from encourage.metrics import NumberMatch
from encourage.prompts import MetaData


class TestNumberMatch(unittest.TestCase):
    def test_exact_numeric_match(self) -> None:
        metric = NumberMatch()
        score, details = metric.compute_score("42", "42")
        self.assertEqual(score, 1.0)
        self.assertEqual(details["type"], "exact_number")
        self.assertTrue(details["matched"])

        score_diff, _ = metric.compute_score("42", "43")
        self.assertEqual(score_diff, 0.0)

        # Decimal format with comma and dot
        score_comma, _ = metric.compute_score("19,5", "19.5")
        self.assertEqual(score_comma, 1.0)

    def test_boolean_match(self) -> None:
        metric = NumberMatch()
        score_yes, _ = metric.compute_score("yes", "yes")
        self.assertEqual(score_yes, 1.0)

        score_ja, _ = metric.compute_score("Ja, das ist zulässig.", "yes")
        self.assertEqual(score_ja, 1.0)

        score_no, _ = metric.compute_score("nein", "yes")
        self.assertEqual(score_no, 0.0)

    def test_free_text_number_recall(self) -> None:
        metric = NumberMatch()
        ref = "Gemäß DIN 18065 beträgt das Maß 14 bis 19 cm."
        pred_full = "Laut DIN 18065 ist ein Bereich von 14 bis 19 cm vorgeschrieben."
        score_full, details_full = metric.compute_score(pred_full, ref)
        self.assertEqual(score_full, 1.0)
        self.assertEqual(details_full["reference_numbers"], ["14", "18065", "19"])
        self.assertEqual(details_full["recall"], 1.0)

        pred_partial = "Die Steigung beträgt 14 cm."
        score_partial, details_partial = metric.compute_score(pred_partial, ref)
        # 1 out of 3 reference numbers matched -> 1/3
        self.assertAlmostEqual(score_partial, 1 / 3, places=4)
        self.assertAlmostEqual(details_partial["recall"], 1 / 3, places=4)

    def test_metric_call_with_response_wrapper(self) -> None:
        responses = ResponseWrapper(
            [
                Response(
                    request_id="1",
                    prompt_id="p1",
                    sys_prompt="system",
                    user_prompt="Wie hoch ist die Steigung?",
                    response="Die Steigung beträgt 19 cm.",
                    meta_data=MetaData(tags={"reference_answer": "19 cm"}),
                ),
                Response(
                    request_id="2",
                    prompt_id="p2",
                    sys_prompt="system",
                    user_prompt="Was ist die Antwort?",
                    response="Der Wert ist 42.",
                    meta_data=MetaData(tags={"reference_answer": "Der Wert ist 43."}),
                ),
            ]
        )

        metric = NumberMatch()
        out = metric(responses)
        self.assertAlmostEqual(out.score, 0.5, places=4)
        self.assertEqual(len(out.raw), 2)
        self.assertEqual(out.raw[0], 1.0)
        self.assertEqual(out.raw[1], 0.0)
        self.assertIn("details", out.misc)

        # Test percentage mode
        metric_pct = NumberMatch(as_percentage=True)
        out_pct = metric_pct(responses)
        self.assertAlmostEqual(out_pct.score, 50.0, places=2)
