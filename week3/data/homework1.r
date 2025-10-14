# 2a preprocessing

# #rpath assisted
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(script_path) > 0) dirname(script_path) else getwd()
data_path <- file.path(script_dir, "wpbc.data")
if (!file.exists(data_path)) {
  alt_path <- file.path(script_dir, "week3", "data", "wpbc.data")
  if (file.exists(alt_path)) {
    data_path <- alt_path
  }
}
if (!file.exists(data_path)) {
  stop("Could not find wpbc.data. Looked in: ", data_path)
}

# wpbc.names column names
col_names <- c("id_number", "outcome", "time", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst", "tumor_size", "lymph_node_status")

# import data, assign column names and if ?, treat as na
if (!file.exists(data_path)) {
  stop("Could not find wpbc.data. Looked in: ", data_path)
}
data <- read.csv(data_path, header = FALSE, col.names = col_names, na.strings = "?")

#  non numeric character to categorical variable
data$outcome <- as.factor(data$outcome)

# print summary stats
print("Summary statistics of df:")

print(summary(data))


# 2b histograms
# 3 plots, radius, mean/se/worst
par(mfrow = c(1, 3))
hist(data$radius_mean, main = "Histogram of Radius Mean", xlab = "Radius Mean")
hist(data$radius_se, main = "Histogram of Radius SE", xlab = "Radius SE")
hist(data$radius_worst, main = "Histogram of Radius Worst", xlab = "Radius Worst")

# reset plot
par(mfrow = c(1, 1))


# 2c scatterplot matrix

# new df w/ ID/Se/Worst column removed
id_col <- which(names(data) == "id_number")
se_cols <- grep("_se", names(data))
worst_cols <- grep("_worst", names(data))
cols_to_remove <- c(id_col, se_cols, worst_cols)
data_small <- data[, -cols_to_remove]

# using pairs w/ numeric columns 
numeric_cols_for_plot <- sapply(data_small, is.numeric)

# scatterplot matrix w/pairs
pairs(data_small[, numeric_cols_for_plot])
pairs(data_small[, numeric_cols_for_plot])
cat("relationship between perimeter, radius area are linear and strong as they both measure size. This is expected as formulas contains the same/similar variables")

# 2d - linear regression
#  perimeter mean as function os radius mean
model <- lm(perimeter_mean ~ radius_mean, data = data)

# plot radius vs perimeter mean
plot(data$radius_mean, data$perimeter_mean,
     main = "Perimeter vs. Radius of Cell Nuclei",
     xlab = "Radius Mean",
     ylab = "Perimeter Mean",
     pch = 19, 
     col = "blue")

# add fitted regression line
abline(model, col = "red", lwd = 2)

# fitted coefficients 
print("Fitted coefficients for the linear model (perimeter_mean ~ radius_mean):")
print(coef(model))
cat("Slope is 6.7 so each additional unit of area increases the perimeter_mean by 6.7 units. The y-intercept is -2.4 and pretty close to zero, which makes sense as the radius approaches zero the perimeter does the same")

# problem 3 loops and conditionals

print("Numbers less than 100 divisible by 3 and 7:")
for (i in 1:99) {
  if (i %% 3 == 0 && i %% 7 == 0) {
    print(i)
  }
}


# 3b -sum of numbers divisible by 3 or 7
total_sum <- 0

# loop through all numbers less than 1000 and divisible by 3 or 7 add them to the total sum 
for (i in 1:999) {
  if (i %% 3 == 0 || i %% 7 == 0) {
    # If it is, add it to the sum
    total_sum <- total_sum + i
  }
}

print("The sum of all numbers less than 1,000 divisible by 3 or 7 is:")
print(total_sum)
