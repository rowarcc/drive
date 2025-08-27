
/*
	join condition should be applied in this order
	1. if there is plan_id and pcp_type, then only have to apply join on the plan_id, pcp_type and service_date
	2. if there is plan_id, then only have to apply join on the plan_id and service_date
	3. otherwise apply join on insco, market, and pcp_type
	4. for the rows in the bec_file that have not find a join in this step, apply join on insco and market, exclude pcp_type from join
	5. apply all the rows that could not did not find its join to insco/market	(TBD)
	
*/
SELECT t.ID,e.gmpi_id,e.elig_month
INTO #mapping
FROM ##transform2 t
JOIN ##elig e ON e.elig_month = LEFT(CONVERT(VARCHAR,t.service_date,112),6)
	AND (t.insco IS NULL OR e.insco = t.insco)
	AND (t.market IS NULL 
				OR e.market = t.market
		OR (t.market = 'DFW' AND e.market IN ('DFWDNG','DFWEMP'))
		OR (t.bec_file = 'osfc' AND t.market = 'TX' AND e.pcp_region IN ('GTX','NTX','ETX','HOU','WTX'))
		OR (t.bec_file = 'osfc' AND t.market = 'FL' AND e.pcp_region IN ('FLO'))
		
	)
	AND (t.plan_id IS NULL 
		OR e.plan_id = t.plan_id
		OR (t.plan_id = 'R6801' AND e.plan_id IN ('R6801-008','R6801-009','R6801-011','R6801-012'))
	)
	AND (t.pcp_type IS NULL OR e.pcp_type = t.pcp_type)
