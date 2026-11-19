# Project adaptation checklist

- [ ] Define the business decision and prediction timestamp.
- [ ] Define the target precisely (e.g. 90+ DPD within next 12 months).
- [ ] Confirm label observation window and outcome maturity.
- [ ] Identify point-in-time source tables and join keys.
- [ ] Remove direct identifiers and post-outcome leakage columns.
- [ ] Decide split strategy: time, customer group, or stratified random.
- [ ] Define baseline and challenger models.
- [ ] Select metrics tied to business costs and operational capacity.
- [ ] Validate calibration if probabilities are used as risk estimates.
- [ ] Assess fairness, stability, explainability and limitations.
- [ ] Complete independent validation and approval before production use.
- [ ] Define monitoring, alert thresholds, retraining and rollback.
