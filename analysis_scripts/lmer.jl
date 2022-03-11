#N.B. the formulas here do not correspond 1:1 with the formulas in the associated
#paper. Firstly, we represent P@10 as a tuple of successes and fails in our actual
#data but refer to it as simply a single variable P@10 in the paper for consistency.
#Secondly the || syntax is bugged in glmmTMB at the moment of writing so we
# write out the uncorrelated formula in full.

using CSV, MixedModels, Statistics, DataFrames, StatsBase, CategoricalArrays
using Plots, GLM, RCall, Distributions

data_loc = "/home/danny/Documents/databases/Flickr_Test/analysis_scripts/lmer.csv"
data = CSV.read(data_loc, DataFrame)

data = data[(data[!, :s_id] .== "s1_1") .| (data[!, :s_id] .== "s2_1"), :]

# set columns to normalise and columns to center
norm_cols = [:duration, :spk_rate, :logword_count, :loglemma_count, :nb_dens,
             :init_cohort]

center_cols = [:n_cons, :n_vowels, :gate]
# turn categorical variables into categorical arrays with the right encoding
data[!, :model_nr] = CategoricalArray(data[!, :model_nr])
data[!, :VQ] = CategoricalArray(data[!, :VQ])
data[!, :lemma] = CategoricalArray(data[!, :lemma])

data[!, :temp] .= 1
data[(data[!, :s_id] .== "s1_1"), :temp] .= -1
data[!, :s_id] = CategoricalArray(data[!, :temp])

data[!, :temp] .= 0
data[(data[!, :word_form] .== "third") .| (data[!, :word_form] .== "plural"), :temp] .= 1
data[(data[!, :word_form] .== "root"), :temp] .= 2
data[!, :word_form_] = CategoricalArray(data[!, :temp])

nouns = data[(data[!, :word_form] .== "singular") .| (data[!, :word_form] .== "plural"),:]

verbs = data[(data[!, :word_form] .== "third") .|
        (data[!, :word_form] .== "participle") .| (data[!, :word_form] .== "root"),:]

# create new dataframes for results after seeing the full word only
nouns_full = nouns[nouns[!,:full_word] .==true, :]
verbs_full = verbs[verbs[!,:full_word] .==true, :]
# normalise and center the variables
for col in norm_cols
    nouns_full[!, col] = (nouns_full[!, col] .- mean(nouns_full[!,col])) ./ std(nouns_full[!,col])

    nouns[!, col] = (nouns[!, col] .- mean(nouns[!,col])) ./ std(nouns[!,col])
    verbs[!, col] = (verbs[!, col] .- mean(verbs[!,col])) ./ std(verbs[!,col])

    verbs_full[!, col] = (verbs_full[!, col] .- mean(verbs_full[!,col])) ./ std(verbs_full[!,col])
end
for col in center_cols
    nouns_full[!, col] = nouns_full[!, col] .- mean(nouns_full[!,col])
    nouns[!, col] = nouns[!, col] .- mean(nouns[!,col])
    verbs[!, col] = verbs[!, col] .- mean(verbs[!,col])
    verbs_full[!, col] = verbs_full[!, col] .- mean(verbs_full[!,col])
end

@rput(nouns, verbs, nouns_full, verbs_full)
R"library(lme4)"
R"library(glmmTMB)"
R"library(car)"
R"library(effects)"

# noun model as presented in paper
R"full_noun_model <- glmmTMB(formula = cbind(success, fail) ~ spk_rate + duration +
                      loglemma_count + (logword_count + word_form_) + VQ + n_vowels +
                      n_cons + s_id  + (1 | lemma)
                      + (0 + VQ|lemma) + (0 + word_form_ | lemma)
                      + (0 + spk_rate | lemma) + (0 + s_id |lemma)
                      + (0 + duration|lemma) + (0 + n_cons | lemma)
                      + (0 + n_vowels | lemma) + (0 + logword_count | lemma)
                      + (1 | model_nr)
                      + (0 + spk_rate | model_nr) + (0 + duration | model_nr)
                      + (0 + n_cons | model_nr) + (0 + logword_count | model_nr)
                      + (0 + n_vowels | model_nr) + (0 + loglemma_count | model_nr)
                      , data = nouns_full, family = betabinomial,
                      REML = TRUE)"

R"summary(full_word_model)"
R"Anova(full_word_model, type = 'II')"
# post hoc models for testing interaction between VQ and word_form/frequency
R"posthoc_noun_model <- glmmTMB(formula = cbind(success, fail) ~ spk_rate + duration +
                        loglemma_count + (logword_count + word_form_) * VQ + n_vowels +
                        n_cons + s_id  + (1 | lemma) + (0 + spk_rate | lemma) +
                        + (0 + duration|lemma) + (0 + n_cons | lemma)
                        + (0 + n_vowels | lemma) + (0 + logword_count | lemma)
                        + (0 + word_form | lemma) + (0 + VQ | lemma)
                        + (1 | model_nr) + (0 + spk_rate | model_nr) + (0 + duration | model_nr)
                        + (0 + n_cons | model_nr) + (0 + logword_count | model_nr)
                        + (0 + n_vowels | model_nr) + (0 + loglemma_count | model_nr)
                        , data = nouns_full, family = betabinomial,
                        REML = FALSE)"

R"summary(posthoc_noun_model)"
R"Anova(posthoc_noun_model, type = 'III')"

R"posthoc_verb_model <- glmmTMB(formula = cbind(success, fail) ~ spk_rate + duration +
                        loglemma_count + (logword_count + word_form_) * VQ + n_vowels +
                        n_cons + s_id  + (1 | lemma) + (0 + spk_rate | lemma) +
                        + (0 + duration|lemma) + (0 + n_cons | lemma)
                        + (0 + n_vowels | lemma) + (0 + logword_count | lemma)
                        + (1 | model_nr)
                        , data = verbs_full, family = betabinomial,
                        REML = TRUE)"

R"summary(posthoc_verb_model)"
R"Anova(posthoc_verb_model, type = 'III')"


# Gating model as presented in the paper
R"gating_noun_model <- glmmTMB(formula = cbind(success, fail) ~
                       (loglemma_count + logword_count) * nb_dens + VQ * gate
                       + init_cohort + s_id +  n_vowels + n_cons +
                       + word_form_ + (1| model_nr)
                       + (1  | lemma) + (0 + nb_dens | lemma) + (0 + gate |lemma)
                       + (0 + init_cohort| lemma) + (0 + s_id | lemma)
                       + (0 + VQ | lemma) + (0 + word_form_ | lemma)
                       + (0 + n_cons | lemma), data = nouns,
                       family = betabinomial, REML = TRUE)"

R"summary(gating_model)"
R"Anova(gating_model, type = 'III')"
