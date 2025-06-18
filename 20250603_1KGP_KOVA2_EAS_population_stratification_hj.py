import hail as hl
import pandas as pd
hl.init(master = 'local[16]', default_reference = "GRCh38", spark_conf={'spark.driver.memory': '64g', 'spark.sql.shuffle.partitions': '200'})

# target data load
vcf_paths = [f"/home/rocky/nfs-mount/exome_snv_subset_n5681_dragen_jointVCF_v1.2.3_folder/dragen_joint_v1.2.3_chrom-{i}.dragen_reheader_n5681_decomposed_snv.vcf.gz" for i in range(1, 26)]
mt = hl.import_vcf(vcf_paths[0], reference_genome='GRCh38', force_bgz=True)

for path in vcf_paths[1:]:
    mt_chr = hl.import_vcf(path, force_bgz=True)
    mt = mt.union_rows(mt_chr)
mt = hl.variant_qc(mt)
mt = mt.filter_rows(mt.variant_qc.call_rate > 0.99)

# 1KGP data load
kgp_mt = hl.read_matrix_table("/home/rocky/nfs-mount/hj/1KGP_phase3_EAS_snp.mt", _n_partitions=300)
kgp_mt = hl.variant_qc(kgp_mt)
kgp_mt = kgp_mt.filter_rows(kgp_mt.variant_qc.call_rate > 0.99)
#sample_to_use = kgp_mt.aggregate_cols(hl.agg.collect(kgp_mt.s))

# kova data load
kova_mt = hl.read_matrix_table("/home/rocky/nfs-mount/hj/kova_n600_old_inbreed_ldprune.mt", _n_partitions=300)

#from gnomad_new.sample_qc.ancestry import apply_onnx_classification_model, assign_population_pcs
#from gnomad_new.utils.filtering import filter_to_adj

# downsample KOVA
import random
sample_to_use = kova_mt.aggregate_cols(hl.agg.collect(kova_mt.s)) # "s" is converted to list object (sample_to_use)
sample_to_use_kova = random.sample(sample_to_use, 100) # random sample 100
kova_mt = kova_mt.filter_cols(hl.literal(sample_to_use_kova).contains(kova_mt.s))
kova_mt.count_cols() # 100 sample

# EAS samples load
eas_samples = hl.import_table("/home/rocky/kova3_eas_v2/20250618_kova3_eas_sampleID_hj.txt", impute = True, key = "sample")

# change key name
eas_samples = eas_samples.key_by(s=eas_samples['sample'])

# remain EAS samples from target mt
target_mt = mt.semi_join_cols(eas_samples)

# check sample size
target_mt.count_cols() # 5798

# define function
intersect_ref = lambda ref_mt, data_mt: (ref_mt.filter_rows(hl.is_defined(data_mt.rows()[ref_mt.row_key])), data_mt.filter_rows(hl.is_defined(ref_mt.rows()[data_mt.row_key])))

# select intersect variants
kova2_kgp_mt, kgp_kova2_mt = intersect_ref(ref_mt=kova_mt, data_mt=kgp_mt)
kova2_kgp_target_mt, target_kova2_kgp_mt = intersect_ref(ref_mt=kova2_kgp_mt, data_mt=target_mt)
target_kova2_kgp_mt, kgp_kova2_target_mt = intersect_ref(ref_mt=target_kova2_kgp_mt, data_mt=kgp_kova2_mt)

# kova2_kgp_target_mt
# kgp_kova2_target_mt
# target_kova2_kgp_mt

# data structure
target_kova2_kgp_structed_mt = target_kova2_kgp_mt.select_entries(target_kova2_kgp_mt.GT, target_kova2_kgp_mt.GQ, target_kova2_kgp_mt.AD)
target_kova2_kgp_structed_mt = target_kova2_kgp_structed_mt.annotate_entries(DP=hl.missing(hl.tint), MIN_DP=hl.missing(hl.tint), PID=hl.missing(hl.tstr), RGQ=hl.missing(hl.tint), SB=hl.missing(hl.tarray(hl.tint32)), PGT=hl.missing(hl.tcall), PL=hl.missing(hl.tarray(hl.tint32)))

kova2_kgp_target_structed_mt = kova2_kgp_target_mt.select_entries(kova2_kgp_target_mt.DP,kova2_kgp_target_mt.GQ, kova2_kgp_target_mt.MIN_DP, kova2_kgp_target_mt.PID, kova2_kgp_target_mt.RGQ, kova2_kgp_target_mt.SB, kova2_kgp_target_mt.GT, kova2_kgp_target_mt.PGT, kova2_kgp_target_mt.AD,kova2_kgp_target_mt.PL)

kgp_kova2_target_structed_mt = kgp_kova2_target_mt.annotate_entries(DP=hl.missing(hl.tint), GQ=hl.missing(hl.tint), MIN_DP=hl.missing(hl.tint), PID=hl.missing(hl.tstr), RGQ=hl.missing(hl.tint), SB=hl.missing(hl.tarray(hl.tint32)), PGT=hl.missing(hl.tcall), AD=hl.missing(hl.tarray(hl.tint32)), PL=hl.missing(hl.tarray(hl.tint32)))

# sort entries
field_order = ['DP', 'GQ', 'MIN_DP', 'PID', 'RGQ', 'SB', 'GT', 'PGT', 'AD', 'PL']
kgp_kova2_target_structed_mt = kgp_kova2_target_structed_mt.select_entries(*field_order)
kova2_kgp_target_structed_mt = kova2_kgp_target_structed_mt.select_entries(*field_order)
target_kova2_kgp_structed_mt = target_kova2_kgp_structed_mt.select_entries(*field_order)

# remove NA18976
kgp_kova2_target_structed_mt = kgp_kova2_target_structed_mt.filter_cols(kgp_kova2_target_structed_mt.s != "NA18976")

# PCA using 1KGP EAS
pca_evals, pca_scores, pca_loadings = hl.hwe_normalized_pca(kgp_kova2_target_structed_mt.GT, k=20, compute_loadings=True) # 여기
pca_mt = kgp_kova2_target_structed_mt.annotate_rows(pca_af=hl.agg.mean(kgp_kova2_target_structed_mt.GT.n_alt_alleles()) / 2)
pca_loadings = pca_loadings.annotate(pca_af=pca_mt.rows()[pca_loadings.key].pca_af)
pca_scores = pca_scores.transmute(**{f'PC{i}': pca_scores.scores[i - 1] for i in range(1, 20 +1)})

sub_tb = hl.import_table("/home/rocky/nfs-mount/hj/20250505_igsr_sample_pop_hj.txt", impute=True, key="Sample name")
sub_tb = sub_tb.annotate(Subpopulation = sub_tb["Population code"])
ref_merge = pca_scores.annotate(Subpopulation = sub_tb[pca_scores.s].Subpopulation)
pca_scores_df = ref_merge.to_pandas()
pca_scores_df.to_csv('/home/rocky/kova3_eas_v2/20250604_1kgp_reference_eas_using_kova2_kova3_PCs_hj.txt', sep = "\t")

# projecting KOVA2 and target data on 1KGP EAS reference PCs
kova2_target = target_kova2_kgp_structed_mt.union_cols(kova2_kgp_target_structed_mt)
data_projection_ht = hl.experimental.pc_project(kova2_target.GT, pca_loadings.loadings, pca_loadings.pca_af)
data_scores = data_projection_ht.transmute(**{f'PC{i}': data_projection_ht.scores[i - 1] for i in range(1, 20+1)})

# merge data
data_ref = ref_merge.union(data_scores, unify=True)

# KOVA2 and KOVA3 annotation
sample_to_use = target_kova2_kgp_structed_mt.aggregate_cols(hl.agg.collect(target_kova2_kgp_structed_mt.s))
data_ref_new = data_ref.annotate(Subpopulation = hl.if_else(~ hl.literal(sample_to_use).contains(data_ref.s) & hl.is_missing(data_ref.Subpopulation), 'KOVA2', data_ref.Subpopulation))

set(data_ref_new.aggregate(hl.agg.collect(data_ref_new.Subpopulation))) # {'CDX', 'CHB', 'CHS', 'JPT', None, 'KOVA', 'KHV'}

data_ref_df = data_ref_new.to_pandas()

data_ref_df.to_csv('/home/rocky/kova3_eas_v2/20250605_1kgp_reference_eas_kova2_kova3_projection_hj.txt', sep = "\t")
