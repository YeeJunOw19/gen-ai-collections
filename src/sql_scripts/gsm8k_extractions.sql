-- CREATE THE ANSWER KEY TABLE IF IT DOES NOT EXISTS
CREATE TABLE IF NOT EXISTS "gen-ai".GSM8KAnswers (
  Id INTEGER,
  ExtractedAnswer INTEGER,
  PRIMARY KEY(Id)
);

-- DELETE OUT OLD DATA FROM ANSWER KEY TABLE
DELETE FROM "gen-ai".GSM8KAnswers;

-- INSERT THE NEWLY LOADED ANSWER KEYS INTO THE TABLE
INSERT INTO "gen-ai".GSM8KAnswers (Id, ExtractedAnswer)
SELECT
    Id,
    CAST(replace(split_part(AnswersGiven, '####', 2), ',', '') AS INT) AS 'ExtractedAnswer'
FROM "gen-ai".RawGSM8KData;