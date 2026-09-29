from api.schemas import TransactionIn

from fraud.serving.schema import REQUIRED_COLUMNS


def test_transaction_in_matches_required_columns() -> None:
    """The API's input contract must never silently drift from the scoring input contract."""
    fields = set(TransactionIn.model_fields) - {"card_id"}
    assert fields == set(REQUIRED_COLUMNS)


def test_transaction_in_requires_card_id() -> None:
    assert TransactionIn.model_fields["card_id"].is_required()
