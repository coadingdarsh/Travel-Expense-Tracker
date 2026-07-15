"""
external_apis.py
=================
Weather context (Open-Meteo — no API key required, free).

Pulls real historical weather for the trip's destination/date so the
end-of-trip debrief can reference actual conditions ("storm that
evening") instead of the traveler having to explain it themselves.
"""

import requests


def _geocode(address: str):
    """Geocode a free-text address via Nominatim (OpenStreetMap, no key required).
    Returns (lat, lon) or None if not found."""
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": "TripAdvocate/1.0 (hackathon demo)"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def check_fare_reasonableness(origin: str, destination: str, amount_paid: float) -> dict:
    """Check whether a transport cost is reasonable for the actual driving
    distance/time between two addresses, using OSRM (routing) and Nominatim
    (geocoding) — both free and keyless.

    Args:
        origin: Pickup address or landmark.
        destination: Drop-off address or landmark.
        amount_paid: What the traveler was charged.

    Returns:
        {"available": bool, "distance_km": float, "duration_min": float,
         "verdict": str, "message": str}
    """
    try:
        origin_coords = _geocode(origin)
        dest_coords = _geocode(destination)

        if not origin_coords or not dest_coords:
            return {
                "available": False, "distance_km": None, "duration_min": None,
                "verdict": "", "message": "Couldn't find one of those locations — try a more specific address.",
            }

        lat1, lon1 = origin_coords
        lat2, lon2 = dest_coords

        route_resp = requests.get(
            f"https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}",
            params={"overview": "false"},
            timeout=10,
        )
        route_resp.raise_for_status()
        route_data = route_resp.json()

        if route_data.get("code") != "Ok":
            return {
                "available": False, "distance_km": None, "duration_min": None,
                "verdict": "", "message": "Could not calculate a route between those points.",
            }

        route = route_data["routes"][0]
        distance_km = route["distance"] / 1000
        duration_min = route["duration"] / 60

        # Rough reasonableness heuristic: typical rideshare rate ~$1.80-$3.20/km
        expected_low  = 4 + (distance_km * 1.8)
        expected_high = 6 + (distance_km * 3.2)

        if amount_paid <= expected_high:
            verdict = f"Reasonable for a {distance_km:.1f}km / {duration_min:.0f}-minute drive."
        else:
            verdict = (
                f"Higher than the typical ${expected_low:.0f}–${expected_high:.0f} range "
                f"for a {distance_km:.1f}km / {duration_min:.0f}-minute drive — "
                "may be worth a note on surge pricing or traffic."
            )

        return {
            "available": True, "distance_km": distance_km, "duration_min": duration_min,
            "verdict": verdict, "message": "",
        }

    except Exception as e:
        return {
            "available": False, "distance_km": None, "duration_min": None,
            "verdict": "", "message": f"Fare check failed: {e}",
        }


def get_weather_context(destination: str, expense_date: str) -> dict:
    """Look up real historical weather for the destination on the given date.
    Fails gracefully — returns available=False if the lookup doesn't succeed.

    Args:
        destination: City name, e.g. "Chicago" or "Chicago, IL".
        expense_date: ISO date string, e.g. "2026-07-15".

    Returns:
        {"available": bool, "summary": str}
    """
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": destination.split(",")[0].strip(), "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return {"available": False, "summary": ""}

        lat, lon = results[0]["latitude"], results[0]["longitude"]

        weather_resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": expense_date,
                "end_date": expense_date,
                "daily": "temperature_2m_max,precipitation_sum,weathercode",
                "timezone": "auto",
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        w = weather_resp.json().get("daily", {})

        if not w.get("temperature_2m_max"):
            return {"available": False, "summary": ""}

        temp_max = w["temperature_2m_max"][0]
        precip   = w["precipitation_sum"][0]

        conditions = []
        if precip and precip > 10:
            conditions.append(f"heavy rain ({precip:.0f}mm)")
        elif precip and precip > 1:
            conditions.append("light rain")
        if temp_max is not None and temp_max > 32:
            conditions.append(f"a heat wave ({temp_max:.0f}°C)")
        elif temp_max is not None and temp_max < 0:
            conditions.append(f"freezing conditions ({temp_max:.0f}°C)")

        if not conditions:
            return {"available": True, "summary": "Weather was unremarkable that day."}

        return {"available": True, "summary": f"There was {' and '.join(conditions)} in {destination} that day."}

    except Exception:
        return {"available": False, "summary": ""}
