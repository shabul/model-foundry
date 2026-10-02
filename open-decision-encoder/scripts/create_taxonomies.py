"""Materialize versioned taxonomy descriptions from source train metadata only."""

import json
from pathlib import Path

from datasets import load_from_disk

MASSIVE = {
    "datetime_query": "Ask for the current date or time",
    "iot_hue_lightchange": "Change the color of a smart light",
    "transport_ticket": "Book or buy a transport ticket",
    "takeaway_query": "Ask about takeaway food options",
    "qa_stock": "Ask for stock market information",
    "general_greet": "Greet the assistant",
    "recommendation_events": "Recommend events to attend",
    "music_dislikeness": "Express dislike for music",
    "iot_wemo_off": "Switch a connected device off",
    "cooking_recipe": "Request a cooking recipe",
    "qa_currency": "Ask about currency exchange",
    "transport_traffic": "Ask about road traffic",
    "general_quirky": "Make a conversational or quirky request",
    "weather_query": "Ask about the weather",
    "audio_volume_up": "Increase audio volume",
    "email_addcontact": "Add an email contact",
    "takeaway_order": "Order takeaway food",
    "email_querycontact": "Look up an email contact",
    "iot_hue_lightup": "Increase smart light brightness",
    "recommendation_locations": "Recommend places to visit",
    "play_audiobook": "Play an audiobook",
    "lists_createoradd": "Create a list or add an item",
    "news_query": "Ask for news",
    "alarm_query": "Ask about existing alarms",
    "iot_wemo_on": "Switch a connected device on",
    "general_joke": "Request a joke",
    "qa_definition": "Ask for a definition",
    "social_query": "Read or search social media",
    "music_settings": "Change music playback settings",
    "audio_volume_other": "Set or query audio volume",
    "calendar_remove": "Remove a calendar event",
    "iot_hue_lightdim": "Decrease smart light brightness",
    "calendar_query": "Ask about calendar events",
    "email_sendemail": "Send an email",
    "iot_cleaning": "Control a cleaning device",
    "audio_volume_down": "Decrease audio volume",
    "play_radio": "Play a radio station",
    "cooking_query": "Ask a cooking question",
    "datetime_convert": "Convert time between time zones",
    "qa_maths": "Calculate a mathematical answer",
    "iot_hue_lightoff": "Turn a smart light off",
    "iot_hue_lighton": "Turn a smart light on",
    "transport_query": "Ask about transport schedules or routes",
    "music_likeness": "Express liking for music",
    "email_query": "Read or search emails",
    "play_music": "Play music",
    "audio_volume_mute": "Mute audio",
    "social_post": "Post to social media",
    "alarm_set": "Set an alarm",
    "qa_factoid": "Ask a factual knowledge question",
    "calendar_set": "Create a calendar event",
    "play_game": "Play a game",
    "alarm_remove": "Remove an alarm",
    "lists_remove": "Remove an item from a list",
    "transport_taxi": "Request a taxi",
    "recommendation_movies": "Recommend movies",
    "iot_coffee": "Control a coffee maker",
    "music_query": "Ask about music",
    "play_podcasts": "Play a podcast",
    "lists_query": "Read or search a list",
}
BANKING_OVERRIDES = {
    "card_arrival": "Ask about a physical card that has not arrived",
    "card_delivery_estimate": "Ask how long card delivery should take",
    "transaction_charged_twice": "Report the same card transaction charged more than once",
    "cash_withdrawal_not_recognised": "Report an unfamiliar ATM withdrawal",
    "card_payment_not_recognised": "Report an unfamiliar card payment",
    "Refund_not_showing_up": "Report an expected refund missing from the balance",
    "reverted_card_payment?": "Ask why a card payment was reversed",
    "extra_charge_on_statement": "Report an additional unexplained statement charge",
    "beneficiary_not_allowed": "A bank transfer recipient is not permitted",
    "balance_not_updated_after_bank_transfer": "A bank transfer arrived but the balance is unchanged",
    "balance_not_updated_after_cheque_or_cash_deposit": "The balance is unchanged after a cash or cheque deposit",
    "transfer_not_received_by_recipient": "The intended recipient has not received a transfer",
    "receiving_money": "Ask how to receive money",
    "get_disposable_virtual_card": "Ask how to obtain a single-use virtual card",
    "getting_virtual_card": "Ask how to obtain a virtual card",
    "get_physical_card": "Ask how to obtain a physical card",
    "order_physical_card": "Request ordering a physical card",
    "getting_spare_card": "Ask how to obtain an additional card",
    "pin_blocked": "A card PIN has been blocked",
    "card_swallowed": "An ATM retained the card",
    "age_limit": "Ask about the minimum or maximum customer age",
    "country_support": "Ask which countries are supported",
    "fiat_currency_support": "Ask which fiat currencies are supported",
}


def main():
    bank = load_from_disk("data/raw/banking77")["train"]
    taxonomy = {}
    for label in sorted(set(bank["category"])):
        readable = label.replace("_", " ").rstrip("?").replace("Refund", "Refund")
        description = BANKING_OVERRIDES.get(label, "The customer asks about " + readable)
        group = next(
            (x for x in ["cash_withdrawal", "top_up", "transfer", "card", "pin", "verify"] if x in label),
            label.split("_")[0],
        )
        taxonomy[label] = {
            "id": label,
            "label": readable.capitalize(),
            "description": description + ".",
            "group": group,
        }
    Path("data/taxonomies/banking77.json").write_text(json.dumps(taxonomy, indent=2) + "\n")
    data = load_from_disk("data/raw/massive")["train"]
    names = data.features["intent"].names
    scenarios = data.features["scenario"].names
    groups = {}
    for row in data:
        groups[names[row["intent"]]] = scenarios[row["scenario"]]
    if set(names) != set(MASSIVE):
        raise ValueError("MASSIVE taxonomy changed")
    taxonomy = {
        name: {"id": name, "label": MASSIVE[name], "description": MASSIVE[name] + ".", "group": groups[name]}
        for name in names
    }
    Path("data/taxonomies/massive.json").write_text(json.dumps(taxonomy, indent=2) + "\n")


if __name__ == "__main__":
    main()
