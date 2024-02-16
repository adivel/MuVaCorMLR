library(shiny)
library(ggplot2)
library(reshape2)
library(shinydashboard)
library(DT)
library(shinyjs)
library(caret)
library(readxl)
library(mlr)

# Function to get numeric columns from a data frame
get_numeric_columns <- function(data) {
  numeric_cols <- sapply(data, is.numeric)
  return(names(numeric_cols[numeric_cols]))
}

# Function to train MLR model
train_mlr_model <- function(data, independent_column, dependent_columns) {
  # Print debug information
  print("Independent Column:")
  print(independent_column)
  print("Dependent Columns:")
  print(dependent_columns)
  print("Column names:")
  print(names(data))
  
  # Check if dependent_columns is a list of column names or a single column name
  if (is.list(dependent_columns)) {
    dependent_columns <- unlist(dependent_columns)
    print("Dependent columns unlisted:")
    print(dependent_columns)
  }
  
  # Create a data frame with selected columns
  selected_data <- data[, c(independent_column, dependent_columns), drop = FALSE]
  
  # Create formula for MLR
  formula <- as.formula(paste(independent_column, "~", paste(dependent_columns, collapse = " + ")))
  
  # Fit MLR model
  mlr_model <- lm(formula, data = selected_data)
  
  return(mlr_model)
}

# Function to compute loss percentage
compute_loss_percentage <- function(model, data, independent_column, dependent_columns) {
  # Predict values
  predictions <- predict(model, data)
  
  # Extract actual values
  actual_values <- data[, independent_column]
  
  # Compute RMSE
  rmse <- RMSE(predictions, actual_values)
  
  # Compute loss percentage
  loss_percentage <- rmse / mean(actual_values) * 100
  
  return(loss_percentage)
}

# UI
ui <- dashboardPage(
  dashboardHeader(title = "MLR Prediction"),
  dashboardSidebar(
    fileInput("file", "Choose File", accept = c(".csv", ".xls", ".xlsx")),
    actionButton("updateBtn", "Update Data"),
    selectInput("download_format", "Choose Download Format: Please wait for tab to load and go to the specific tab to download the image.", choices = c("png", "jpeg", "pdf", "svg"), selected = "png"),
    downloadButton("download_dynamic_plot", "Download Dynamic Heatmap", style = "color: #FFFFFF; background-color: #007BFF; border-color: #007BFF"),
    downloadButton("download_threshold_plot", "Download Threshold Heatmap", style = "color: #FFFFFF; background-color: #DC3545; border-color: #DC3545")
  ),
  dashboardBody(
    useShinyjs(),
    tabsetPanel(
      tabPanel("View Dataset",
               verbatimTextOutput("view_columns"),
               DTOutput("view_data")),
      tabPanel("Dynamic Heatmap",
               uiOutput("column_selector"),
               sliderInput("sizeSlider", "Adjust Size", min = 0.5, max = 2, value = 1, step = 0.1),
               plotOutput("plot_dynamic_heatmap")
      ),
      tabPanel("Threshold Heatmap",
               uiOutput("column_selector_threshold"),
               sliderInput("thresholdSlider", "Threshold", min = -1, max = 1, value = 0, step = 0.01),
               plotOutput("plot_threshold_heatmap")
      ),
      tabPanel("Training your multi-linear regression",
               uiOutput("mlr_input"),
               actionButton("run_mlr_prediction", "Run MLR Prediction"),
               verbatimTextOutput("mlr_output")
      ),
      tabPanel("Predict New Values",
               uiOutput("predict_input"),
               actionButton("predict_button", "Predict"),
               verbatimTextOutput("prediction_output")
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Reactive values for storing data and MLR model
  data <- reactiveVal(NULL)
  mlr_model <- reactiveVal(NULL)
  
  # Load CSV or Excel data
  load_data <- function(file_path) {
    if (grepl("\\.csv$", file_path, ignore.case = TRUE)) {
      return(read.csv(file_path))
    } else if (grepl("\\.xls$|\\.xlsx$", file_path, ignore.case = TRUE)) {
      return(read_excel(file_path))
    } else {
      stop("Unsupported file format. Please upload a CSV or Excel file.")
    }
  }
  
  observeEvent(input$file, {
    new_data <- load_data(input$file$datapath)
    if (!is.null(data())) {
      data(merge(data(), new_data, all = TRUE))
    } else {
      data(new_data)
    }
  })
  
  observeEvent(input$updateBtn, {
    new_data <- load_data(input$file$datapath)
    if (!is.null(data())) {
      data(merge(data(), new_data, all = TRUE))
    } else {
      data(new_data)
    }
  })
  
  # View Columns
  output$view_columns <- renderPrint({
    req(data())
    colnames(data())
  })
  
  # View Dataset
  output$view_data <- renderDT({
    req(data())
    datatable(data())
  })
  
  # Column selector
  output$column_selector <- renderUI({
    req(data())
    numeric_columns <- get_numeric_columns(data())
    checkboxGroupInput("heatmap_columns", "Select Numeric Columns for Heatmap", choices = numeric_columns, selected = numeric_columns)
  })
  
  # Dynamic Heatmap
  output$plot_dynamic_heatmap <- renderPlot({
    req(data(), input$heatmap_columns)
    selected_columns <- input$heatmap_columns
    selected_data <- data()[, selected_columns, drop = FALSE]
    
    cor_matrix <- cor(selected_data, use = "complete.obs")
    melted_data <- melt(cor_matrix)
    
    gg <- ggplot(melted_data, aes(x = Var1, y = Var2, fill = value)) +
      geom_tile() +
      scale_fill_distiller(palette = "Spectral", breaks = seq(-1, 1, by = 0.2), limits = c(-1, 1)) +
      labs(title = "Dynamic Correlation Heatmap",
           x = "Variable 1",
           y = "Variable 2") +
      theme(
        axis.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90, hjust = 1),
        axis.text.y = element_text(hjust = 1, size = 8),
        panel.grid = element_blank()
      ) +
      coord_fixed(ratio = input$sizeSlider, xlim = c(0.5, ncol(cor_matrix) + 0.5), ylim = c(0.5, nrow(cor_matrix) + 0.5)) +
      scale_x_discrete(expand = c(0.002, 0.002)) +
      scale_y_discrete(expand = c(0.002, 0.002))
    
    print(gg)
  })
  
  # Enable the download button for dynamic heatmap
  shinyjs::enable("download_dynamic_plot")
  
  # Download handler for Dynamic Heatmap
  output$download_dynamic_plot <- downloadHandler(
    filename = function() {
      paste("dynamic_heatmap", ".", input$download_dynamic_plot, ".", input$download_format, sep = "")
    },
    content = function(file) {
      req(data(), input$heatmap_columns)  # Ensure data and input are available
      selected_columns <- input$heatmap_columns
      selected_data <- data()[, selected_columns, drop = FALSE]
      cor_matrix <- cor(selected_data, use = "complete.obs")
      melted_data <- melt(cor_matrix)
      
      gg <- ggplot(melted_data, aes(x = Var1, y = Var2, fill = value)) +
        geom_tile() +
        scale_fill_distiller(palette = "Spectral", breaks = seq(-1, 1, by = 0.2), limits = c(-1, 1)) +
        labs(title = "Dynamic Correlation Heatmap",
             x = "Variable 1",
             y = "Variable 2") +
        theme(
          axis.text = element_text(size = 8),
          axis.text.x = element_text(angle = 90, hjust = 1),
          axis.text.y = element_text(hjust = 1, size = 8),
          panel.grid = element_blank()
        ) +
        coord_fixed(ratio = input$sizeSlider, xlim = c(0.5, ncol(cor_matrix) + 0.5), ylim = c(0.5, nrow(cor_matrix) + 0.5)) +
        scale_x_discrete(expand = c(0.002, 0.002)) +
        scale_y_discrete(expand = c(0.002, 0.002))
      
      ggsave(file, gg, width = 8, height = 6, units = "in", device = input$download_format)
    }
  )
  
  # Column selector for threshold heatmap
  output$column_selector_threshold <- renderUI({
    req(data())
    numeric_columns_threshold <- get_numeric_columns(data())
    checkboxGroupInput("heatmap_columns_threshold", "Select Numeric Columns for Threshold Heatmap", choices = numeric_columns_threshold, selected = numeric_columns_threshold)
  })
  
  # Threshold Heatmap
  output$plot_threshold_heatmap <- renderPlot({
    req(data(), input$heatmap_columns_threshold)
    selected_columns_threshold <- input$heatmap_columns_threshold
    selected_data_threshold <- data()[, selected_columns_threshold, drop = FALSE]
    
    cor_matrix_threshold <- cor(selected_data_threshold, use = "complete.obs")
    melted_data_threshold <- melt(cor_matrix_threshold)
    
    # Set the threshold
    threshold <- input$thresholdSlider
    
    # Create a new variable indicating whether the correlation is above the threshold
    data_with_threshold <- transform(melted_data_threshold, above_threshold = value >= threshold)
    
    gg <- ggplot(data_with_threshold, aes(x = Var1, y = Var2, fill = above_threshold)) +
      geom_tile() +
      geom_text(aes(label = round(value, 2)), vjust = 1) +
      scale_fill_manual(values = c("FALSE" = "grey", "TRUE" = "red"), breaks = c(FALSE, TRUE), guide = FALSE) +
      labs(title = "Threshold Correlation Heatmap",
           x = "Variable 1",
           y = "Variable 2") +
      theme(
        axis.text = element_text(size = 8),
        axis.text.x = element_text(angle = 90, hjust = 1),
        axis.text.y = element_text(hjust = 1, size = 8),
        panel.grid = element_blank()
      ) +
      coord_fixed(ratio = 1, xlim = c(0.5, ncol(cor_matrix_threshold) + 0.5), ylim = c(0.5, nrow(cor_matrix_threshold) + 0.5)) +
      scale_x_discrete(expand = c(0.002, 0.002)) +
      scale_y_discrete(expand = c(0.002, 0.002))
    
    print(gg)
  })
  
  # Enable the download button for threshold heatmap
  shinyjs::enable("download_threshold_plot")
  
  ################################### TRAINING ##########################################################################################
  
  # MLR Prediction tab
  output$mlr_input <- renderUI({
    req(data())
    numeric_columns_mlr <- get_numeric_columns(data())
    tagList(
      numericInput("num_dependent_values", 
                   "Enter the Number of Dependent Values", 
                   value = min(2, length(numeric_columns_mlr)), min = 1, max = length(numeric_columns_mlr)),
      selectInput("mlr_independent_column", 
                  "Select Independent Variable to be Predicted", 
                  choices = numeric_columns_mlr),
      uiOutput("dependent_columns_ui")
    )
  })
  
  output$dependent_columns_ui <- renderUI({
    req(input$num_dependent_values)
    numeric_columns_mlr <- get_numeric_columns(data())
    
    # Exclude the selected independent variable from the dependent variable choices
    dependent_choices <- setdiff(numeric_columns_mlr, input$mlr_independent_column)
    
    # Dynamic UI for selecting dependent variables
    dependent_columns <- lapply(1:input$num_dependent_values, function(i) {
      selectInput(paste0("mlr_dependent_column_", i), 
                  label = paste("Select Value for Dependent Variable", i),
                  choices = dependent_choices, selected = dependent_choices[1])
    })
    
    tagList(dependent_columns)
  })
  
  observeEvent(input$run_mlr_prediction, {
    output$mlr_output <- renderPrint({
      req(data(), input$num_dependent_values, input$mlr_independent_column)
      
      num_dependent_values <- as.integer(input$num_dependent_values)
      
      # Extract values for each dependent variable
      dependent_columns <- lapply(1:num_dependent_values, function(i) input[[paste0("mlr_dependent_column_", i)]])
      
      # Train MLR model
      mlr_model_val <- train_mlr_model(data(), input$mlr_independent_column, dependent_columns)
      mlr_model(mlr_model_val)  # Save MLR model in reactiveVal
      
      # Compute loss percentage
      loss_percentage <- compute_loss_percentage(mlr_model_val, data(), input$mlr_independent_column, dependent_columns)
      
      return(paste("Loss Percentage:", round(loss_percentage, 2), "%"))
    })
  })
  
  # Predict New Values tab
  output$predict_input <- renderUI({
    req(data())
    numeric_columns_mlr <- get_numeric_columns(data())
    tagList(
      selectInput("predict_independent_column", 
                  "Select Independent Variable to Predict",
                  choices = numeric_columns_mlr),
      uiOutput("predict_dependent_columns")
    )
  })
  
  output$predict_dependent_columns <- renderUI({
    req(input$predict_independent_column)
    numeric_columns_mlr <- get_numeric_columns(data())
    dependent_choices <- setdiff(numeric_columns_mlr, input$predict_independent_column)
    lapply(dependent_choices, function(column) {
      numericInput(paste0("predict_", column), 
                   label = paste("Enter Value for", column), 
                   value = 0)
    })
  })
  
  observeEvent(input$predict_button, {
    output$prediction_output <- renderPrint({
      req(data(), input$predict_independent_column)
      independent_column <- input$predict_independent_column
      
      # Prepare new data for prediction
      new_data <- data.frame(stringsAsFactors = FALSE)
      new_data[[independent_column]] <- NA
      
      numeric_columns_mlr <- get_numeric_columns(data())
      dependent_columns <- setdiff(numeric_columns_mlr, input$predict_independent_column)
      for (column in dependent_columns) {
        new_data[[column]] <- input[[paste0("predict_", column)]]
      }
      
      # Predict new values
      predictions <- predict(mlr_model(), newdata = new_data)
      
      return(paste("Predicted Value for", independent_column, ":", round(predictions, 2)))
    })
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
