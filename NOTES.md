This repo predicts soccer home, draw, and away results and over/under goal totals. It is not an MLB or NBA model.

The football-data.org pull for Premier League season 2025 was called once on 2026-09-30: https://api.football-data.org/v4/competitions/PL/matches?season=2025
It returned HTTP 403 and denied access:
{"message":"The resource you are looking for is restricted and apparently not within your permissions. Please check your subscription.","errorCode":403}
No matches were invented. No completed-game file was written. The SoccerDataLoader data branch was not retried. This branch was not pushed.
