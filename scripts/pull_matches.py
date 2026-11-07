"""Pull completed soccer matches from football-data.co.uk through 2026-11-07.

The modelling files keep the SoccerDataLoader columns. Scores, shots, cards,
corners, and odds are copied from the public match files. Team form is the
mean of earlier matches in the same league, division, and season. Market
average and market maximum use the published Avg/Max columns when the file
has them, otherwise the mean and max of the bookmaker columns in that file.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2026-11-07")
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "soccer" / "modelling"
BASE = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DRAW_MARGIN = 0.25
ROLLING = 3
START_YEARS = range(2019, 2027)

LEAGUES = {
    "England": ("E", ("0", "1", "2", "3", "C")),
    "Scotland": ("SC", ("0", "1", "2", "3")),
    "Germany": ("D", ("1", "2")),
    "Italy": ("I", ("1", "2")),
    "Spain": ("SP", ("1", "2")),
    "France": ("F", ("1", "2")),
    "Netherlands": ("N", ("1",)),
    "Belgium": ("B", ("1",)),
    "Portugal": ("P", ("1",)),
    "Turkey": ("T", ("1",)),
    "Greece": ("G", ("1",)),
}

COLUMNS = [
    "date",
    "league",
    "division",
    "year",
    "home_team",
    "away_team",
    "odds__market_maximum__home_win__full_time_goals",
    "odds__market_maximum__draw__full_time_goals",
    "odds__market_maximum__away_win__full_time_goals",
    "odds__market_maximum__over_2.5__full_time_goals",
    "odds__market_maximum__under_2.5__full_time_goals",
    "odds__market_average__home_win__full_time_goals",
    "odds__market_average__draw__full_time_goals",
    "odds__market_average__away_win__full_time_goals",
    "odds__market_average__over_2.5__full_time_goals",
    "odds__market_average__under_2.5__full_time_goals",
    "home__points__avg",
    "home__adj_points__avg",
    "home__goals_for__avg",
    "home__goals_against__avg",
    "home__adj_goals_for__avg",
    "home__adj_goals_against__avg",
    "home__points__latest_avg",
    "home__adj_points__latest_avg",
    "home__goals_for__latest_avg",
    "home__goals_against__latest_avg",
    "home__adj_goals_for__latest_avg",
    "home__adj_goals_against__latest_avg",
    "away__points__avg",
    "away__adj_points__avg",
    "away__goals_for__avg",
    "away__goals_against__avg",
    "away__adj_goals_for__avg",
    "away__adj_goals_against__avg",
    "away__points__latest_avg",
    "away__adj_points__latest_avg",
    "away__goals_for__latest_avg",
    "away__goals_against__latest_avg",
    "away__adj_goals_for__latest_avg",
    "away__adj_goals_against__latest_avg",
    "target__home_team__full_time_goals",
    "target__away_team__full_time_goals",
    "target__home_team__shots",
    "target__away_team__shots",
    "target__home_team__shots_on_target",
    "target__away_team__shots_on_target",
    "target__home_team__corners",
    "target__away_team__corners",
    "target__home_team__fouls_committed",
    "target__away_team__fouls_committed",
    "target__home_team__yellow_cards",
    "target__away_team__yellow_cards",
    "target__home_team__red_cards",
    "target__away_team__red_cards",
]

FORM_STATS = [
    "points",
    "adj_points",
    "goals_for",
    "goals_against",
    "adj_goals_for",
    "adj_goals_against",
]
HOME_BOOKS = ["B365H", "BWH", "IWH", "PSH", "WHH", "VCH", "LBH", "GBH", "SBH", "SJH", "SYH", "SOH"]
DRAW_BOOKS = ["B365D", "BWD", "IWD", "PSD", "WHD", "VCD", "LBD", "GBD", "SBD", "SJD", "SYD", "SOD"]
AWAY_BOOKS = ["B365A", "BWA", "IWA", "PSA", "WHA", "VCA", "LBA", "GBA", "SBA", "SJA", "SYA", "SOA"]
OVER_BOOKS = ["B365>2.5", "P>2.5", "GB>2.5"]
UNDER_BOOKS = ["B365<2.5", "P<2.5", "GB<2.5"]


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def division_number(league: str, code: str) -> int:
    codes = LEAGUES[league][1]
    if "0" in codes or "C" in codes:
        return 5 if code == "C" else int(code) + 1
    return int(code)


def download(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "sports-betting/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} {url}")
        return None
    except urllib.error.URLError as exc:
        print(f"FAIL {url} {exc.reason}")
        return None


def read_frame(content: bytes) -> pd.DataFrame | None:
    if content[:200].lstrip().lower().startswith(b"<"):
        return None
    text = content.decode("utf-8-sig", errors="replace")
    frame = pd.read_csv(io.StringIO(text), encoding="utf-8", on_bad_lines="skip")
    frame = frame.rename(columns=lambda name: str(name).replace("\ufeff", "").strip())
    if "HomeTeam" not in frame.columns and "Home" in frame.columns:
        frame = frame.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam"})
    if "FTHG" not in frame.columns and "HG" in frame.columns:
        frame = frame.rename(columns={"HG": "FTHG", "AG": "FTAG"})
    if not {"HomeTeam", "AwayTeam", "Date"}.issubset(frame.columns):
        return None
    return frame


def parse_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, dayfirst=True, format="mixed", errors="coerce")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def market(frame: pd.DataFrame, published: list[str], books: list[str], how: str) -> pd.Series:
    for column in published:
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().any():
                return values
    present = [column for column in books if column in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    numbers = frame[present].apply(pd.to_numeric, errors="coerce")
    if how == "max":
        return numbers.max(axis=1, skipna=True)
    return numbers.mean(axis=1, skipna=True)


def add_adjusted(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    long["adj_goals_for"] = long["goals_for"].astype(float)
    long["adj_goals_against"] = long["goals_against"].astype(float)
    for source_for, source_against, scale in (
        ("red_for", "red_against", 1.0),
        ("yellow_for", "yellow_against", 1 / 20),
        ("sot_for", "sot_against", 1 / 10 + 1 / 50),
        ("corners_for", "corners_against", 1 / 100),
    ):
        if source_for in long.columns and long[source_for].notna().any():
            long["adj_goals_for"] = long["adj_goals_for"] + long[source_for] * scale
            long["adj_goals_against"] = long["adj_goals_against"] + long[source_against] * scale
    goals_for = long["goals_for"]
    goals_against = long["goals_against"]
    long["points"] = np.where(goals_for > goals_against, 3, np.where(goals_for == goals_against, 1, 0)).astype(float)
    adjusted_for = long["adj_goals_for"]
    adjusted_against = long["adj_goals_against"]
    missing = adjusted_for.isna() | adjusted_against.isna()
    decided = np.where(
        adjusted_for > adjusted_against + DRAW_MARGIN,
        3,
        np.where(np.abs(adjusted_for - adjusted_against) <= DRAW_MARGIN, 1, 0),
    ).astype(float)
    long["adj_points"] = np.where(missing, np.nan, decided)
    return long


def prior_mean(values: pd.Series, window: int | None) -> pd.Series:
    if window is None:
        return values.expanding().mean().shift(1)
    return values.rolling(window, min_periods=1).mean().shift(1)


def after_mean(values: pd.Series, window: int | None) -> pd.Series:
    if window is None:
        return values.expanding().mean()
    return values.rolling(window, min_periods=1).mean()


def team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["home_team"],
            "goals_for": matches["target__home_team__full_time_goals"],
            "goals_against": matches["target__away_team__full_time_goals"],
            "sot_for": matches["target__home_team__shots_on_target"],
            "sot_against": matches["target__away_team__shots_on_target"],
            "corners_for": matches["target__home_team__corners"],
            "corners_against": matches["target__away_team__corners"],
            "yellow_for": matches["target__home_team__yellow_cards"],
            "yellow_against": matches["target__away_team__yellow_cards"],
            "red_for": matches["target__home_team__red_cards"],
            "red_against": matches["target__away_team__red_cards"],
        }
    )
    away = pd.DataFrame(
        {
            "date": matches["date"],
            "team": matches["away_team"],
            "goals_for": matches["target__away_team__full_time_goals"],
            "goals_against": matches["target__home_team__full_time_goals"],
            "sot_for": matches["target__away_team__shots_on_target"],
            "sot_against": matches["target__home_team__shots_on_target"],
            "corners_for": matches["target__away_team__corners"],
            "corners_against": matches["target__home_team__corners"],
            "yellow_for": matches["target__away_team__yellow_cards"],
            "yellow_against": matches["target__home_team__yellow_cards"],
            "red_for": matches["target__away_team__red_cards"],
            "red_against": matches["target__home_team__red_cards"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long = long.dropna(subset=["date", "team", "goals_for", "goals_against"])
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    return add_adjusted(long)


def attach_form(matches: pd.DataFrame, long: pd.DataFrame, side: str) -> pd.DataFrame:
    team_col = f"{side}_team"
    form = long[["team", "date", *FORM_STATS]].copy()
    for stat in FORM_STATS:
        form[f"{side}__{stat}__avg"] = form.groupby("team", sort=False)[stat].transform(lambda values: prior_mean(values, None))
        form[f"{side}__{stat}__latest_avg"] = form.groupby("team", sort=False)[stat].transform(
            lambda values: prior_mean(values, ROLLING)
        )
    keep = ["team", "date"] + [column for column in form.columns if column.startswith(f"{side}__")]
    renamed = form[keep].rename(columns={"team": team_col})
    return matches.merge(renamed, on=["date", team_col], how="left")


def form_lookup(long: pd.DataFrame) -> pd.DataFrame:
    form = long[["team", "date", *FORM_STATS]].sort_values(["date", "team"]).copy()
    for stat in FORM_STATS:
        form[f"{stat}__avg"] = form.groupby("team", sort=False)[stat].transform(lambda values: after_mean(values, None))
        form[f"{stat}__latest_avg"] = form.groupby("team", sort=False)[stat].transform(
            lambda values: after_mean(values, ROLLING)
        )
    keep = ["team", "date"] + [column for column in form.columns if column.endswith("avg")]
    return form[keep].sort_values("date")


def attach_fixture_form(fixtures: pd.DataFrame, lookup: pd.DataFrame, side: str) -> pd.DataFrame:
    if fixtures.empty or lookup.empty:
        return fixtures
    team_col = f"{side}_team"
    right = lookup.rename(columns={"team": team_col})
    right = right.rename(columns={column: f"{side}__{column}" for column in right.columns if column.endswith("avg")})
    left = fixtures.sort_values("date")
    merged = pd.merge_asof(
        left,
        right.sort_values("date"),
        on="date",
        by=team_col,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.sort_index()


def prepare_matches(frame: pd.DataFrame, league: str, division: int, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["date"] = parse_dates(work["Date"])
    work["home_team"] = work["HomeTeam"].astype(str).str.strip()
    work["away_team"] = work["AwayTeam"].astype(str).str.strip()
    work = work.dropna(subset=["date"])
    work = work[(work["home_team"] != "") & (work["away_team"] != "") & (work["home_team"] != "nan")]
    goals_home = numeric(work, "FTHG")
    goals_away = numeric(work, "FTAG")
    played = goals_home.notna() & goals_away.notna() & (work["date"] <= AS_OF)
    upcoming = goals_home.isna() & goals_away.isna() & (work["date"] >= AS_OF)
    base = pd.DataFrame(
        {
            "date": work["date"],
            "league": league,
            "division": division,
            "year": year,
            "home_team": work["home_team"],
            "away_team": work["away_team"],
            "odds__market_maximum__home_win__full_time_goals": market(work, ["MaxH", "BbMxH"], HOME_BOOKS, "max"),
            "odds__market_maximum__draw__full_time_goals": market(work, ["MaxD", "BbMxD"], DRAW_BOOKS, "max"),
            "odds__market_maximum__away_win__full_time_goals": market(work, ["MaxA", "BbMxA"], AWAY_BOOKS, "max"),
            "odds__market_maximum__over_2.5__full_time_goals": market(work, ["Max>2.5", "BbMx>2.5"], OVER_BOOKS, "max"),
            "odds__market_maximum__under_2.5__full_time_goals": market(work, ["Max<2.5", "BbMx<2.5"], UNDER_BOOKS, "max"),
            "odds__market_average__home_win__full_time_goals": market(work, ["AvgH", "BbAvH"], HOME_BOOKS, "mean"),
            "odds__market_average__draw__full_time_goals": market(work, ["AvgD", "BbAvD"], DRAW_BOOKS, "mean"),
            "odds__market_average__away_win__full_time_goals": market(work, ["AvgA", "BbAvA"], AWAY_BOOKS, "mean"),
            "odds__market_average__over_2.5__full_time_goals": market(work, ["Avg>2.5", "BbAv>2.5"], OVER_BOOKS, "mean"),
            "odds__market_average__under_2.5__full_time_goals": market(work, ["Avg<2.5", "BbAv<2.5"], UNDER_BOOKS, "mean"),
            "target__home_team__full_time_goals": goals_home,
            "target__away_team__full_time_goals": goals_away,
            "target__home_team__shots": numeric(work, "HS"),
            "target__away_team__shots": numeric(work, "AS"),
            "target__home_team__shots_on_target": numeric(work, "HST"),
            "target__away_team__shots_on_target": numeric(work, "AST"),
            "target__home_team__corners": numeric(work, "HC"),
            "target__away_team__corners": numeric(work, "AC"),
            "target__home_team__fouls_committed": numeric(work, "HF"),
            "target__away_team__fouls_committed": numeric(work, "AF"),
            "target__home_team__yellow_cards": numeric(work, "HY"),
            "target__away_team__yellow_cards": numeric(work, "AY"),
            "target__home_team__red_cards": numeric(work, "HR"),
            "target__away_team__red_cards": numeric(work, "AR"),
        }
    )
    completed = base.loc[played].drop_duplicates(["date", "home_team", "away_team"])
    scheduled = base.loc[upcoming].drop_duplicates(["date", "home_team", "away_team"])
    return completed.copy(), scheduled.copy()


def with_form(played: pd.DataFrame, upcoming: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if played.empty:
        return played, upcoming
    long = team_rows(played)
    modeled = attach_form(played, long, "home")
    modeled = attach_form(modeled, long, "away")
    if upcoming.empty:
        return modeled, upcoming
    lookup = form_lookup(long)
    fixtures = attach_fixture_form(upcoming, lookup, "home")
    fixtures = attach_fixture_form(fixtures, lookup, "away")
    return modeled, fixtures


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    for column in COLUMNS:
        if column not in output.columns:
            output[column] = np.nan
    output = output[COLUMNS].sort_values(["date", "home_team", "away_team"])
    output = output.drop_duplicates(["date", "home_team", "away_team"])
    output["date"] = pd.to_datetime(output["date"]).dt.strftime("%d/%m/%Y")
    output.to_csv(path, index=False)


def ending_year(stamp: pd.Timestamp) -> int:
    return int(stamp.year + 1 if stamp.month >= 7 else stamp.year)


def division_codes() -> dict[str, tuple[str, int]]:
    mapping = {}
    for league, (prefix, codes) in LEAGUES.items():
        for code in codes:
            mapping[f"{prefix}{code}"] = (league, division_number(league, code))
    return mapping


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: dict[tuple[str, int, int], pd.DataFrame] = {}
    for league, (prefix, codes) in LEAGUES.items():
        for code in codes:
            division = division_number(league, code)
            for start_year in START_YEARS:
                year = start_year + 1
                url = BASE.format(season=season_code(start_year), code=f"{prefix}{code}")
                content = download(url)
                time.sleep(0.12)
                if not content:
                    continue
                raw = read_frame(content)
                if raw is None or raw.empty:
                    print(f"skip {url}")
                    continue
                downloaded[(league, division, year)] = raw
                print(f"downloaded {league} {division} {year} rows {len(raw)}")

    fixture_frames = []
    fixtures_content = download(FIXTURES_URL)
    if fixtures_content:
        raw_fixtures = read_frame(fixtures_content)
        codes = division_codes()
        if raw_fixtures is not None and "Div" in raw_fixtures.columns:
            print(f"fixtures feed rows {len(raw_fixtures)}")
            for division_code, part in raw_fixtures.groupby("Div"):
                if division_code not in codes:
                    continue
                league, division = codes[str(division_code)]
                part = part.copy()
                part["date"] = parse_dates(part["Date"])
                goals_home = numeric(part, "FTHG")
                goals_away = numeric(part, "FTAG")
                completed = part[goals_home.notna() & goals_away.notna() & part["date"].notna() & (part["date"] <= AS_OF)]
                if not completed.empty:
                    grouped_year = completed["date"].map(ending_year)
                    for year, rows in completed.groupby(grouped_year):
                        key = (league, division, int(year))
                        extra = rows.drop(columns=["date"])
                        downloaded[key] = pd.concat([downloaded.get(key, pd.DataFrame()), extra], ignore_index=True)
                        print(f"fixtures results added {league} {division} {int(year)} rows {len(extra)}")
                played_probe, upcoming = prepare_matches(part, league, division, 0)
                del played_probe
                if not upcoming.empty:
                    upcoming["year"] = upcoming["date"].map(ending_year)
                    fixture_frames.append(upcoming)
        else:
            print("fixtures feed missing")
    else:
        print("fixtures feed missing")

    written = []
    upcoming_frames = []
    latest_long: dict[tuple[str, int], tuple[int, pd.DataFrame]] = {}
    for (league, division, year), raw in sorted(downloaded.items()):
        played, upcoming = prepare_matches(raw, league, division, year)
        played, upcoming = with_form(played, upcoming)
        if played.empty:
            print(f"no completed rows {league} {division} {year}")
            continue
        path = OUT_DIR / f"{league.lower()}_{division}_{year}.csv"
        write_frame(path, played)
        written.append((path.name, len(played), played["date"].min(), played["date"].max()))
        print(f"wrote {path.name} rows {len(played)} {played['date'].min().date()} {played['date'].max().date()}")
        latest_long[(league, division)] = (year, team_rows(played))
        if not upcoming.empty:
            upcoming_frames.append(upcoming)

    if fixture_frames:
        external = pd.concat(fixture_frames, ignore_index=True)
        kept = []
        for (league, division), part in external.groupby(["league", "division"]):
            if (league, division) not in latest_long:
                continue
            season_year, long = latest_long[(league, division)]
            part = part[part["year"] == season_year].copy()
            if part.empty:
                continue
            lookup = form_lookup(long)
            part = attach_fixture_form(part, lookup, "home")
            part = attach_fixture_form(part, lookup, "away")
            kept.append(part)
        if kept:
            upcoming_frames.append(pd.concat(kept, ignore_index=True))

    if upcoming_frames:
        fixtures = pd.concat(upcoming_frames, ignore_index=True)
        write_frame(OUT_DIR / "fixtures.csv", fixtures)
        print(f"fixtures rows {len(fixtures)}")
    else:
        print("fixtures rows 0")
    if not written:
        raise SystemExit("no match files were written")
    total = sum(rows for _, rows, _, _ in written)
    minimum = min(start for _, _, start, _ in written)
    maximum = max(end for _, _, _, end in written)
    print(f"files {len(written)}")
    print(f"rows {total}")
    print(f"min date {minimum.date()}")
    print(f"max date {maximum.date()}")


if __name__ == "__main__":
    main()
