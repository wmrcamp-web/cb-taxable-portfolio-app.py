# CB Taxable Portfolio

### How to run it on your own machine

Prerequisite: install `uv` if you don't already have it.

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Sync the dependencies

   ```
   $ uv sync
   ```

2. Run the app

   ```
   $ uv run streamlit run streamlit_app.py
   ```

### Deploy publicly with Streamlit Community Cloud

The app can be hosted at a public `streamlit.app` URL. The repository includes
the Streamlit entry point (`streamlit_app.py`) and the locked dependencies
(`pyproject.toml` and `uv.lock`). The app does not require deployment secrets.

1. Commit and push the app files to the `main` branch of
   `wmrcamp-web/cb-taxable-portfolio-app.py`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with the
   GitHub account that can access the repository, then choose **Create app**.
3. Select repository `wmrcamp-web/cb-taxable-portfolio-app.py`, branch `main`,
   and entry point `streamlit_app.py`.
4. In **Advanced settings**, select Python 3.14 to match `.python-version`, and
   choose an available custom app subdomain if desired.
5. Deploy. Community Cloud will provide a public URL and update the app when
   changes are pushed to the selected branch.

This deployment is public and does not require visitors to sign in. Do not
enter sensitive portfolio information into a public app. Keep local holdings
data and credentials out of Git commits.

#### Preload taxable holdings with Streamlit secrets

To preload holdings without committing them to the repository, open the app's
settings in Streamlit Community Cloud, select **Secrets**, and add a
`HOLDINGS_CSV` multiline string:

```toml
HOLDINGS_CSV = """
TICKER,ALLOCATION_PCT,GAIN_PCT,TERM
AAPL,15,25,LT
"""
```

Use the same CSV columns and `LT`/`ST` values accepted by the app. Save the
secret and redeploy or restart the app. If the secret is not configured, the
app uses its built-in example holding instead. Secrets keep the data out of
GitHub, but because this app is currently public, anyone who can use the app
may still see the preloaded holdings in the editable field or results.

### Deploy to Streamlit in Snowflake

The Snowflake project definition is in `snowflake.yml`. The database and schema
are set to `USER$.PUBLIC`, and the selected compute pool and query warehouse are
`SYSTEM_COMPUTE_POOL_CPU` and `COMPUTE_WH`. The app uses the Python 3.11
container runtime. Before deploying, configure an external access integration
that allows the HTTPS hosts listed below.

The app needs a compute pool that supports Streamlit container apps and a
warehouse for queries. Snowflake privileges and network rules may need to be
configured by an account administrator; compute resources can incur Snowflake
usage charges.

Attach Snowflake's shared PyPI artifact repository
(`snowflake.snowpark.pypi_shared_repository`) when creating/deploying the app,
and ensure the deployment role has the required repository usage grant. This
repository provides Python packages and is separate from external network
access. The app also fetches market data from Yahoo Finance and the S&P 500
constituent list from Wikipedia, so its external access integration must allow
the required HTTPS hosts for those services.

The Snowflake CLI connection name prepared for this account is
`cb-taxable-portfolio`. Once the external access integration is configured and
the connection is authenticated, deploy from the repository root:

```
REPLACE_WITH_EXTERNAL_ACCESS_INTEGRATION
```

Only the source files explicitly listed in `snowflake.yml` are deployed.
Local secrets and `holdings.csv` are not included.
