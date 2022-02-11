from snorkel.labeling import labeling_function, LFAnalysis, PandasLFApplier, LabelingFunction
ABSTAIN = -1


def social_science_based_LFs(df_train, df_val):
	NONTROUBLING = 0
	TROUBLING = 1
	@labeling_function()
	def lf_S2502_C05_001E_label(x):
		if x.S2502_C05_001E >= x.S2502_C05_001E_max_threshold:
			return TROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2502_C05_018E_label(x):
		if x.S2502_C05_018E >= x.S2502_C05_018E_max_threshold:
			return TROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2502_C06_019E_label(x):
		if x.S2502_C06_019E >= x.S2502_C06_019E_max_threshold:
			return NONTROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2502_C06_020E_label(x):
		if x.S2502_C06_020E >= x.S2502_C06_020E_max_threshold:
			return NONTROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2502_C06_021E_label(x):
		if x.S2502_C06_021E >= x.S2502_C06_021E_max_threshold:
			return NONTROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2001_C02_003E_label(x):
		if x.S2001_C02_003E >= x.S2001_C02_003E_max_threshold:
			return NONTROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2503_C05_028E_label(x):
		if x.S2503_C05_028E >= x.S2503_C05_028E_max_threshold:
			return TROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S2503_C05_045E_label(x):
		if x.S2503_C05_045E >= x.S2503_C05_045E_max_threshold:
			return TROUBLING 
		else:
			return ABSTAIN

	@labeling_function()
	def lf_S1702_C02_018E_label(x):
		if x.S1702_C02_018E >= x.S1702_C02_018E_max_threshold:
			return NONTROUBLING 
		else:
			return ABSTAIN

	lfs = [lf_S2502_C05_001E_label,
				lf_S2502_C05_018E_label,
				lf_S2502_C06_019E_label,
				lf_S2502_C06_020E_label,
				lf_S2502_C06_021E_label,
				lf_S2001_C02_003E_label,
				lf_S2503_C05_028E_label,
				lf_S2503_C05_045E_label,
				lf_S1702_C02_018E_label]
	applier = PandasLFApplier(lfs=lfs)
	L_train = applier.apply(df=df_train)
	L_val = applier.apply(df=df_val)
	print(LFAnalysis(L=L_val, lfs=lfs).lf_summary())
	return L_train, L_val

