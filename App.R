library(shiny)
library(ggplot2)
library(reshape2)
library(shinydashboard)
library(DT)
library(shinyjs)
library(caret)

# Define UI
ui <- dashboardPage(
  dashboardHeader(title = "Correlation Heatmap Shiny App"),
  dashboardSidebar(
    fileInput("file", "Choose CSV File", accept = c(".csv")),
    actionButton("updateBtn", "Update Data"),
    selectInput("download_format", "Choose Download Format", choices = c("png", "jpeg", "pdf", "svg"), selected = "png"),
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
      tabPanel("Multilinear Regression",
               uiOutput("independent_variable_selector"),
               actionButton("submitBtn", "Predict"),
               textOutput("predicted_value")
      )
    )
  )
)

# Define server
server <- function(input, output, session) {

  # Reactive values for storing data
  data <- reactiveVal(NULL)

  # Load CSV data
  observeEvent(input$file, {
    new_data <- read.csv(input$file$datapath)
    if (!is.null(data())) {
      data(merge(data(), new_data, all = TRUE))
    } else {
      data(new_data)
    }
  })

  # Update data button
  observeEvent(input$updateBtn, {
    new_data <- read.csv(input$file$datapath)
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
    checkboxGroupInput("heatmap_columns", "Select Columns for Heatmap", choices = colnames(data()), selected = colnames(data()))
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
      coord_fixed(ratio = input$sizeSlider, xlim = c(0.5, ncol(cor_matrix) + 0.5), ylim = c(0.5, ncol(cor_matrix) + 0.5)) +
      scale_x_discrete(expand = c(0.002, 0.002)) +
      scale_y_discrete(expand = c(0.002, 0.002))

    print(gg)
  })

  # Column selector for threshold heatmap
  output$column_selector_threshold <- renderUI({
    req(data())
    checkboxGroupInput("heatmap_columns_threshold", "Select Columns for Threshold Heatmap", choices = colnames(data()), selected = colnames(data()))
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
      coord_fixed(ratio = 1, xlim = c(0.5, ncol(cor_matrix_threshold) + 0.5), ylim = c(0.5, ncol(cor_matrix_threshold) + 0.5)) +
      scale_x_discrete(expand = c(0.002, 0.002)) +
      scale_y_discrete(expand = c(0.002, 0.002))

    print(gg)
  })

  # Enable the download button for threshold heatmap
  shinyjs::enable("download_threshold_plot")

  # Download handler for Threshold Heatmap
  output$download_threshold_plot <- downloadHandler(
    filename = function() {
      paste("threshold_heatmap", input$download_threshold_plot, ".", input$download_format, sep = "")
    },
    content = function(file) {
      req(data(), input$heatmap_columns_threshold, input$thresholdSlider)  # Ensure data and input are available
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
        coord_fixed(ratio = 1, xlim = c(0.5, ncol(cor_matrix_threshold) + 0.5), ylim = c(0.5, ncol(cor_matrix_threshold) + 0.5)) +
        scale_x_discrete(expand = c(0.002, 0.002)) +
        scale_y_discrete(expand = c(0.002, 0.002))

      ggsave(file, gg, width = 8, height = 6, units = "in", device = input$download_format)
    }
  )

  # Column selector for independent variables
  output$independent_variable_selector <- renderUI({
    req(data())
    checkboxGroupInput("independent_variables", "Select Independent Variables", choices = colnames(data()), selected = colnames(data()))
  })

  # Predicted value
  output$predicted_value <- renderText({
    req(data(), input$independent_variables, input$submitBtn)
    independent_variables <- input$independent_variables
    selected_data <- data()[, c(independent_variables), drop = FALSE]
    model <- lm(selected_data[, 1] ~ ., data = selected_data[, -1, drop = FALSE])
    new_data <- data.frame(tail(selected_data[, -1, drop = FALSE], 1))
    predicted_value <- predict(model, newdata = new_data)
    paste("Predicted Value: ", round(predicted_value, 2))
  })
}



# Run the application
shinyApp(ui = ui, server = server)

