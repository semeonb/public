QUERY = """ WITH pl AS
(
  SELECT
  x.idContacto,
  x.regDate,
  REPLACE(REGEXP_REPLACE(IFNULL(partner, 'missing'), r'[^\x00-\x7F]+', ''), ' ', 'missing') AS partner,
  REPLACE(REGEXP_REPLACE(IFNULL(gender, 'missing'), r'[^\x00-\x7F]+', '') , ' ', 'missing') AS gender,
  REPLACE(REGEXP_REPLACE(IFNULL(ResidentCountry, 'missing'), r'[^\x00-\x7F]+', '') , ' ', 'missing') AS country,
  REPLACE(REGEXP_REPLACE(IFNULL(regChannel, 'missing'), r'[^\x00-\x7F]+', '') , ' ', 'missing') AS regChannel,
  REPLACE(REGEXP_REPLACE(IFNULL(mexosAdvertiser, 'missing'), r'[^\x00-\x7F]+', '') , ' ', 'missing') AS advertiser,
  CONCAT('weekday_', CAST(EXTRACT(DAYOFWEEK FROM regDate ) AS STRING)) AS cat_regweekday,
  TIMESTAMP_DIFF(TIMESTAMP('{statsdate}'), x.regDate, DAY) AS daysSinceReg,
  TIMESTAMP_DIFF(TIMESTAMP('{statsdate}'), x.firstDepositDate, DAY) AS daysSinceFd
  FROM `warehouse.dimPlayers` x
  WHERE testAccount = 'No'
  AND x.regDate < TIMESTAMP('{statsdate}')
  AND x.tDepositAmt > 0
  AND x.firstDepositDate < TIMESTAMP('{statsdate}')
  AND regDate IS NOT NULL
  AND X.firstDepositDate > '1900-01-01'
),
casino AS
(
  WITH data AS
  (
  SELECT
  idContacto,
  statsdate,
  bets,
  wins,
  gameCategory
  FROM
    (
      SELECT
      idContacto,
      localDt AS statsdate,
      CASE
        WHEN LOWER(gameCategory) LIKE '%slot%' THEN 'slots'
        WHEN LOWER(gameCategory) LIKE '%ruleta%' OR LOWER(gameCategory) LIKE '%roulette%' THEN 'roulette'
        WHEN LOWER(gameCategory) LIKE '%blackjack%' OR LOWER(gameCategory) LIKE '%black jack%' THEN 'blackjack'
        ELSE 'other'
      END AS gameCategory,
      SUM(betAmountBaseCur) bets,
      SUM(winAmountBaseCur) wins
      FROM `warehouse.casinoAgg`
      WHERE
      localDt >= DATE_SUB(DATE('{statsdate}'), INTERVAL {days_back} DAY)
      AND localDt <= DATE('{statsdate}')
      AND betAmountBaseCur + winAmountBaseCur > 0
      GROUP BY 1, 2, 3
    )
  ),
  total AS
  (
  SELECT
  idContacto AS id,
  SUM(bets) AS bets,
  SUM(wins) AS wins,
  STRING_AGG(DISTINCT gameCategory) AS gamesList
  FROM data
  GROUP BY 1
  ),
  sd AS
  (
  SELECT
  idContacto AS id,
  SUM(bets) AS bets,
  SUM(wins) AS wins,
  STRING_AGG(DISTINCT gameCategory) AS gamesList
  FROM data
  WHERE 
  statsdate >= DATE_SUB(DATE('{statsdate}'), INTERVAL {days_back} DAY) AND
  statsdate <= DATE('{statsdate}')
  GROUP BY 1
  )
  SELECT
  total.id AS idContacto,
  IFNULL(total.bets, 0) casino_bets,
  IFNULL(total.wins, 0) casino_wins,
  IFNULL(total.bets, 0) - IFNULL(total.wins, 0) AS casino_gw,
  IFNULL(total.gamesList, 'missing') AS casino_game_list
  FROM total
),
hs AS
(
  WITH data AS
  (
    SELECT
    idContacto,
    CASE
      WHEN channel like '%mobile%' THEN 'mobile'
      WHEN channel like '%desktop%' THEN 'desktop'
      WHEN channel like '%retail%' THEN 'retail'
      ELSE 'missing'
    END AS channel,
    statshour,
    casinoNGRbaseCur,
    sportsbookNGRbaseCur,
    depositAmountBaseCur,
    withdrawAmountBaseCur,
    betAmountBaseCur AS casino_bet,
    ticketTurnoverBaseCur AS sports_bet,
    localStatshour,
    IF(channel='EMAIL', promo_open, 0) AS promo_emails_opened,
    IF(channel='SMS', promo_click, 0) AS promo_sms_clicked,
    IF(channel='EMAIL', promo_click, 0) AS promo_emails_clicked
    FROM `warehouse.hourlystats`
    WHERE
    localStatshour >= TIMESTAMP_SUB(TIMESTAMP('{statsdate}'), INTERVAL 24*{days_back} HOUR)
    AND localStatshour <= TIMESTAMP('{statsdate}')
  ),
  total AS
  (
    SELECT
    idContacto AS id,
    SUM(IFNULL(casinoNGRbaseCur, 0)) ngr_casino,
    SUM(IFNULL(sportsbookNGRbaseCur, 0)) ngr_sports,
    SUM(IFNULL(depositAmountBaseCur, 0)) deposit,
    SUM(IFNULL(withdrawAmountBaseCur, 0)) withdraw,
    COUNT(DISTINCT CASE WHEN casino_bet + sports_bet > 0 THEN statshour END) AS active_hours,
    MAX(CASE WHEN casino_bet + sports_bet > 0
        AND TIMESTAMP_DIFF(TIMESTAMP('{statsdate}'), statshour, HOUR) = 0 THEN 1 ELSE 0 END) AS activeLastHour,
    COUNT(DISTINCT CASE WHEN withdrawAmountBaseCur > 0 THEN statshour END) AS deposit_hours,
    STRING_AGG(DISTINCT channel) AS channelList,
    EXTRACT(HOUR FROM MAX(localStatshour)) AS lastLocalHour,
    SUM(promo_emails_opened) promo_emails_opened,
    SUM(promo_sms_clicked) promo_sms_clicked,
    SUM(promo_emails_clicked) promo_emails_clicked
    FROM data
    GROUP BY 1
   )
   SELECT
   total.id AS idContacto,
   IFNULL(ngr_casino, 0) ngr_casino,
   IFNULL(ngr_sports, 0) ngr_sports,
   IFNULL(deposit, 0) deposit,
   IFNULL(withdraw ,0) withdraw,
   IFNULL(active_hours, 0) active_hours,
   IFNULL(deposit_hours, 0) deposit_hours,
   IFNULL(total.channelList, 'missing') channelList,
   activeLastHour,
   lastLocalHour,
   IFNULL(promo_emails_opened, 0) promo_emails_opened,
   IFNULL(promo_sms_clicked, 0) promo_sms_clicked,
   IFNULL(promo_emails_clicked, 0) promo_emails_clicked
   FROM total
),
sports AS
(
  WITH data AS
  (
   SELECT
   id,
   ts,
   turnoverBaseCur AS turnover,
   sport,
   isLive,
   betType,
   leagueName
   FROM
     (
     SELECT
      idContacto AS id,
      ts,
      CASE
        WHEN (LOWER(sportName) LIKE '%football%' OR LOWER(sportName) LIKE '%soccer%') AND LOWER(sportName) NOT LIKE '%1025083526%' THEN 'football'
        WHEN LOWER(sportName) LIKE '%tennis%' THEN 'tennis'
        WHEN LOWER(sportName) LIKE '%baseball%' THEN 'baseball'
        WHEN LOWER(sportName) LIKE '%basketball%' THEN 'basketball'
        WHEN LOWER(sportName) LIKE '%greyhound%' THEN 'greyhounds'
        WHEN LOWER(sportName) LIKE '%horse%' THEN 'horses'
        WHEN LOWER(sportName) LIKE '%hockey%' THEN 'hockey'
        WHEN LOWER(sportName) LIKE '%rugby%' THEN 'rugby'
        WHEN LOWER(sportName) LIKE '%1025083526%' THEN 'american_football'
        WHEN LOWER(sportName) LIKE '%box%' THEN 'boxing'
        WHEN LOWER(sportName) LIKE '%volleyball%' THEN 'volleyball'
      ELSE 'other'
      END AS sport,
      turnoverBaseCur,
      isLive,
      betType,
      leagueName
      FROM `warehouse.sportsAgg`
      WHERE
      statsdate > TIMESTAMP_SUB(TIMESTAMP('{statsdate}'), INTERVAL 24*{days_back} HOUR)
      AND statsdate <= TIMESTAMP('{statsdate}')
      AND turnoverBaseCur > 0
     )
   ),
  total AS
  (
   SELECT
   data.id,
   SUM(data.turnover) turnover,
   STRING_AGG(DISTINCT sport) AS sportsList,
   STRING_AGG(DISTINCT isLive) AS isLiveList,
   STRING_AGG(DISTINCT betType) AS betTypeList
   FROM data
   GROUP BY 1
  ),
  leagues AS
  (
    SELECT
    x.id,
    COUNT(futureLeagueName) AS futureLeaguesCnt
    FROM
    (
      SELECT
      x.id,
      x.leagueName
      FROM data x
      GROUP BY 1, 2
    ) x
    LEFT JOIN
    (
      SELECT DISTINCT LEAGUENAME AS futureLeagueName FROM warehouse.factBgtEvents
      WHERE
      DEV_STARTDATE >= TIMESTAMP('{statsdate}')
      AND DEV_STARTDATE <= TIMESTAMP_ADD(TIMESTAMP('{statsdate}'), INTERVAL {churn_period} * 24 HOUR)
    ) y
    ON x.leagueName = y.futureLeagueName
    GROUP BY 1
  )
  SELECT
  total.id idContacto,
  IFNULL(total.turnover,0) sports_turnover,
  IFNULL(total.sportsList, 'missing') AS sportsList,
  IFNULL(total.isLiveList, 'missing') isLiveList,
  IFNULL(total.betTypeList, 'missing') betTypeList,
  IFNULL(leagues.futureLeaguesCnt, 0) futureLeaguesCnt
  FROM total
  LEFT JOIN leagues ON total.id = leagues.id
),
data AS
(
 SELECT
 pl.idContacto,
 IFNULL(hs.active_hours, 0) active_hours,
 IFNULL(hs.channelList, 'missing') channelList,
 IFNULL(hs.deposit_hours, 0) deposit_hours,
 IFNULL(hs.deposit, 0) deposit,
 IFNULL(hs.ngr_sports, 0) ngr_sports,
 IFNULL(hs.ngr_casino, 0) ngr_casino,
 IFNULL(sports.betTypeList, 'missing') betTypeList,
 IFNULL(hs.withdraw, 0) withdraw,
 IFNULL(casino.casino_wins, 0) casino_wins,
 IFNULL(sports.sportsList, 'missing') sportsList,
 IFNULL(casino.casino_game_list, 'missing') casino_game_list,
 IFNULL(casino.casino_bets, 0) casino_bets,
 IFNULL(casino.casino_gw, 0) casino_gw,
 IFNULL(sports.futureLeaguesCnt, 0) futureLeaguesCnt,
 IFNULL(sports.isLiveList, 'missing') isLiveList,
 IFNULL(sports.sports_turnover, 0) sports_turnover,
 pl.daysSinceReg,
 IFNULL(hs.activeLastHour, 0) activeLastHour,
 IFNULL(lastLocalHour, 25) lastLocalHour,
 IFNULL(promo_emails_opened, 0) promo_emails_opened,
 IFNULL(promo_sms_clicked, 0) promo_sms_clicked,
 IFNULL(promo_emails_clicked, 0) promo_emails_clicked
 FROM
 pl 
 INNER JOIN hs ON pl.idContacto = hs.idContacto
 LEFT JOIN casino ON pl.idContacto = casino.idContacto
 LEFT JOIN sports ON pl.idContacto = sports.idContacto
),
pred AS
(
  SELECT
  idContacto,
  SUM(depositAmountBaseCur) AS depositAmountBaseCur,
  SUM(betAmountBaseCur) AS casino_bet,
  SUM(ticketTurnoverBaseCur) AS sports_bet
  FROM
  `warehouse.hourlystats`
  WHERE
  localStatshour <= TIMESTAMP_SUB(TIMESTAMP('{statsdate}'), INTERVAL 24*{churn_period} HOUR)
  AND localStatshour > TIMESTAMP('{statsdate}')
  GROUP BY 1
)
SELECT
{days_back} AS days_back,
{churn_period} AS churn_period,
TIMESTAMP('{statsdate}') AS statsdate,
pl.idContacto,
pl.partner,
CASE
  WHEN LOWER(pl.gender) = 'hombre' THEN 'M'
  WHEN LOWER(pl.gender) = 'mujer' THEN 'F'
  ELSE pl.gender
END AS gender,
pl.country,
pl.regChannel,
pl.advertiser,
pl.cat_regweekday,
pl.daysSinceReg,
pl.daysSinceFd,
data.channelList,
data.sportsList,
data.active_hours,
data.deposit_hours,
data.deposit,
data.ngr_sports,
data.ngr_casino,
data.betTypeList,
data.withdraw,
data.casino_wins,
data.casino_game_list,
data.casino_bets,
data.casino_gw,
data.futureLeaguesCnt,
data.isLiveList,
data.sports_turnover,
data.activeLastHour,
data.lastLocalHour,
pred.depositAmountBaseCur AS pred_deposit_base_cur,
pred.casino_bet AS pred_casino_bet,
pred.sports_bet aS pred_sports_bet,
data.promo_emails_opened,
data.promo_sms_clicked,
data.promo_emails_clicked
FROM pl
INNER JOIN data ON pl.idContacto = data.idContacto
LEFT JOIN pred ON pl.idContacto = pred.idContacto """
