bec_file	service_date	insco	market	plan_id	plan_type	pcp_type	IP_ABE_INCURRED	IP_ABE_CLAIMS_LAG	IP_RESERVE	OP_ABE_INCURRED	OP_ABE_CLAIMS_LAG	OP_RESERVE	PRO_ABE_INCURRED	PRO_ABE_CLAIMS_LAG	PRO_RESERVE	Dental_ABE_INCURRED	Dental_ABE_CLAIMS_LAG	Dental_RESERVE	ID
dental	2025-07-01	NULL	NULL	H2509-002	NULL	NULL	NULL	NULL	NULL	NULL	NULL	NULL	NULL	NULL	NULL	41177.28	12227	28950.28	3194
fl	2022-10-01	MED	MIAMI	NULL	SNP	Employed	18155.44	18155.36	0.08	69997.36	69995.51	1.85	71363.13	71362.88	0.25	NULL	NULL	NULL	3205



gmpi_Id	member_id	elig_month	insco	group_id	contract_id	pbp	plan_id	pcp_region	pcp_market	pcp_type	ika_clinic_code	ofl_flag	snp_flag
702394470	4136972	202107	UGT	RTX	H1278	010	H1278-010	SWTX	VLY	Affiliated	VLYAFF	NULL	Non-SNP
701485887	3465764	202107	UGT	RGV	H4527	013	H4527-013	SWTX	VLY	Affiliated	VLYAFF	NULL	Non-SNP
676472055	5794986	202303	CST	ECS	H2228	041	H2228-041	SWTX	ELP	Employed	ELPWT	N	SNP
676470663	2701897	202303	UGT	ENM	H2228	023	H2228-023	SWTX	ELP	Employed	ELPWT	N	Non-SNP


DROP TABLE IF EXISTS #pmpm_output;

SELECT t.ID,t.bec_file,t.service_date,t.insco,t.market,t.plan_id,t.plan_type,t.pcp_type
		,CAST(SUM(t.IP_ABE_INCURRED) / COUNT(*) AS DECIMAL(18,2)) AS IP_INCURRED_PMPM
		,CAST(SUM(t.IP_RESERVE) / COUNT(*) AS DECIMAL(18,2)) AS IP_RESERVE_PMPM
		,CAST(SUM(t.OP_ABE_INCURRED) / COUNT(*) AS DECIMAL(18,2)) AS OP_ABE_INCURRED_PMPM
		,CAST(SUM(t.OP_RESERVE) / COUNT(*) AS DECIMAL(18,2)) AS OP_RESERVE_PMPM
		,CAST(SUM(t.PRO_ABE_INCURRED) / COUNT(*) AS DECIMAL(18,2)) AS PRO_ABE_INCURRED_PMPM
		,CAST(SUM(t.PRO_RESERVE) / COUNT(*) AS DECIMAL(18,2)) AS PRO_RESERVE_PMPM
		,CAST(SUM(t.Dental_ABE_INCURRED) / COUNT(*) AS DECIMAL(18,2)) AS Dental_ABE_INCURRED_PMPM
		,CAST(SUM(t.Dental_RESERVE) / COUNT(*) AS DECIMAL(18,2)) AS Dental_RESERVE_PMPM
INTO #pmpm_output
FROM ##transform2 t
JOIN ##elig e ON e.elig_month = LEFT(CONVERT(VARCHAR,t.service_date,112),6)
WHERE (t.insco IS NULL OR e.insco = t.insco)
	AND (t.market IS NULL OR e.pcp_market = t.market)
	AND (t.plan_id IS NULL OR e.plan_id = t.plan_id)
	AND (t.pcp_type IS NULL OR e.pcp_type = t.pcp_type)
GROUP BY t.ID,t.bec_file,t.service_date,t.insco,t.market,t.plan_id,t.plan_type,t.pcp_type


DROP TABLE IF EXISTS #output_table;
SELECT t.ID, e.*
INTO #output_table
FROM ##elig e
JOIN ##transform2 t ON e.elig_month = LEFT(CONVERT(VARCHAR,t.service_date,112),6)
WHERE (t.insco IS NULL OR e.insco = t.insco)
	AND (t.market IS NULL OR e.pcp_market = t.market)
	AND (t.plan_id IS NULL OR e.plan_id = t.plan_id)
	AND (t.pcp_type IS NULL OR e.pcp_type = t.pcp_type)


SELECT*FROM ##transform2 WHERE ID NOT IN (SELECT ID FROM #pmpm_output)
