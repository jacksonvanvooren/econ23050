library(readr)
library(dplyr)
library(ggplot2)

# Read TSV files
df_patent <- read_tsv("/Users/jacksonvanvooren/Downloads/g_patent.tsv")
df_assignee <- read_tsv("/Users/jacksonvanvooren/Downloads/g_assignee_disambiguated.tsv")

# Filter out missing organizations
df_assignee_filtered <- df_assignee %>%
  filter(disambig_assignee_organization != "NA")

# Merge data and convert date
merged_df <- merge(df_assignee_filtered, df_patent[, c("patent_id", "patent_date")], by = "patent_id", all.x = TRUE)
merged_df$patent_date <- as.Date(merged_df$patent_date)

# Filter patents after 2005
merged_df_filtered <- merged_df %>%
  filter(format(patent_date, "%Y") > 2005)

# Identify top 10 companies by number of patents
company_patent_counts <- merged_df_filtered %>%
  group_by(disambig_assignee_organization) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  top_n(10, count)

# Define the top companies
big_companies <- c(
  "SAMSUNG ELECTRONICS CO., LTD.",
  "International Business Machines Corporation",
  "Canon Kabushiki Kaisha",
  "LG ELECTRONICS INC.",
  "Sony Group Corporation",
  "Intel Corporation",
  "QUALCOMM Incorporated",
  "Apple Inc.",
  "TAIWAN SEMICONDUCTOR MANUFACTURING COMPANY LTD.",
  "KABUSHIKI KAISHA TOSHIBA"
)

# Filter only patents from big companies
merged_df_big <- merged_df_filtered %>%
  filter(disambig_assignee_organization %in% big_companies)

merged_df_cleaned <- merged_df_big %>%
  mutate(year = format(patent_date, "%Y")) %>%
  select(disambig_assignee_organization, year)

# Compute total number of patents per year
total_patents_per_year <- merged_df_cleaned %>%
  group_by(year) %>%
  summarise(total_patents = n(), .groups = "drop")

# Compute patent count per company per year
patent_count_by_company_year <- merged_df_cleaned %>%
  group_by(disambig_assignee_organization, year) %>%
  summarise(patent_count = n(), .groups = "drop") %>%
  left_join(total_patents_per_year, by = "year") %>%
  mutate(patent_share = patent_count / total_patents) %>%
  filter(year >= 2005 & year <= 2020) %>%
  arrange(disambig_assignee_organization, year)

# Plot patent share over time
ggplot(patent_count_by_company_year, aes(x = year, y = patent_share, color = disambig_assignee_organization, group = disambig_assignee_organization)) +
  geom_line() +
  geom_point() +
  labs(title = "Patent Share by Company (2006-2020)",
       x = "Year",
       y = "Patent Share",
       color = "Company") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right",
    legend.justification = "left",
    legend.box = "vertical",
    legend.margin = margin(10, 10, 10, 10)
  ) +
  guides(color = guide_legend(ncol = 1))
