-- ============================================
-- MemeShield Database Schema
-- MySQL Database for Hate Speech Detection
-- ============================================

-- Create database
CREATE DATABASE IF NOT EXISTS memeshield
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE memeshield;

-- ============================================
-- Main Tables
-- ============================================

-- Memes table: Stores uploaded memes and their analysis results
CREATE TABLE IF NOT EXISTS memes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- File information
    filename VARCHAR(255) NOT NULL COMMENT 'Unique filename stored on server',
    original_filename VARCHAR(255) COMMENT 'Original filename uploaded by user',
    file_size INT COMMENT 'File size in bytes',
    file_type VARCHAR(50) COMMENT 'MIME type of the file',
    
    -- OCR extracted text
    extracted_text TEXT COMMENT 'Raw text extracted using OCR',
    processed_text TEXT COMMENT 'Preprocessed/cleaned text',
    
    -- Prediction results
    prediction VARCHAR(50) NOT NULL COMMENT 'hate or non-hate',
    confidence FLOAT NOT NULL COMMENT 'Confidence score (0-1)',
    
    -- Detailed scores (for future use)
    text_score FLOAT COMMENT 'Score from text analysis only',
    image_score FLOAT COMMENT 'Score from image analysis only',
    fusion_score FLOAT COMMENT 'Combined multimodal score',
    
    -- Metadata
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INT COMMENT 'Time taken to process in milliseconds',
    model_version VARCHAR(50) DEFAULT '1.0' COMMENT 'Version of AI model used',
    
    -- Indexes
    INDEX idx_prediction (prediction),
    INDEX idx_timestamp (timestamp),
    INDEX idx_confidence (confidence)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================
-- Additional Tables (For Future Extensions)
-- ============================================

-- Users table: For user management (future scope)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('user', 'moderator', 'admin') DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    is_active BOOLEAN DEFAULT TRUE,
    
    INDEX idx_email (email),
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Analysis logs: Detailed logging for debugging and monitoring
CREATE TABLE IF NOT EXISTS analysis_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    meme_id INT,
    step_name VARCHAR(100) NOT NULL COMMENT 'OCR, preprocessing, CNN, fusion, etc.',
    status ENUM('started', 'completed', 'failed') NOT NULL,
    details TEXT COMMENT 'Additional details or error messages',
    duration_ms INT COMMENT 'Duration of this step',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (meme_id) REFERENCES memes(id) ON DELETE CASCADE,
    INDEX idx_meme_id (meme_id),
    INDEX idx_step (step_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Feedback table: For user feedback on predictions
CREATE TABLE IF NOT EXISTS feedback (
    id INT AUTO_INCREMENT PRIMARY KEY,
    meme_id INT NOT NULL,
    user_feedback ENUM('correct', 'incorrect') NOT NULL,
    suggested_label VARCHAR(50) COMMENT 'User suggested correct label',
    comment TEXT COMMENT 'Additional feedback from user',
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (meme_id) REFERENCES memes(id) ON DELETE CASCADE,
    INDEX idx_meme_id (meme_id),
    INDEX idx_feedback (user_feedback)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Model versions: Track different model versions and their performance
CREATE TABLE IF NOT EXISTS model_versions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    version VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    accuracy FLOAT COMMENT 'Validation accuracy',
    f1_score FLOAT COMMENT 'F1 score on test set',
    deployed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    model_path VARCHAR(500) COMMENT 'Path to saved model files',
    
    INDEX idx_version (version),
    INDEX idx_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Categories/Tags: For categorizing types of hate speech (future scope)
CREATE TABLE IF NOT EXISTS hate_categories (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    severity_level INT DEFAULT 1 COMMENT '1-5 severity scale',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- Meme-Category mapping (many-to-many)
CREATE TABLE IF NOT EXISTS meme_categories (
    meme_id INT NOT NULL,
    category_id INT NOT NULL,
    confidence FLOAT COMMENT 'Confidence for this category',
    
    PRIMARY KEY (meme_id, category_id),
    FOREIGN KEY (meme_id) REFERENCES memes(id) ON DELETE CASCADE,
    FOREIGN KEY (category_id) REFERENCES hate_categories(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================
-- Sample Data for Testing
-- ============================================

-- Insert default model version
INSERT INTO model_versions (version, description, accuracy, f1_score, is_active) VALUES
('1.0.0', 'Initial release - MobileNetV2 + LSTM fusion model', 0.85, 0.82, TRUE);

-- Insert hate speech categories
INSERT INTO hate_categories (name, description, severity_level) VALUES
('Racism', 'Content targeting race or ethnicity', 5),
('Sexism', 'Content targeting gender', 4),
('Religious', 'Content targeting religious groups', 4),
('Homophobia', 'Content targeting LGBTQ+ community', 4),
('Ableism', 'Content targeting disabilities', 3),
('General Hate', 'General hateful content', 3);


-- ============================================
-- Useful Queries
-- ============================================

-- Get statistics
-- SELECT 
--     COUNT(*) as total_memes,
--     SUM(CASE WHEN prediction = 'hate' THEN 1 ELSE 0 END) as hate_count,
--     SUM(CASE WHEN prediction = 'non-hate' THEN 1 ELSE 0 END) as non_hate_count,
--     AVG(confidence) as avg_confidence
-- FROM memes;

-- Get recent predictions
-- SELECT * FROM memes ORDER BY timestamp DESC LIMIT 10;

-- Get predictions by date
-- SELECT DATE(timestamp) as date, COUNT(*) as count 
-- FROM memes 
-- GROUP BY DATE(timestamp) 
-- ORDER BY date DESC;


-- ============================================
-- Stored Procedures
-- ============================================

DELIMITER //

-- Procedure to get dashboard statistics
CREATE PROCEDURE IF NOT EXISTS GetDashboardStats()
BEGIN
    SELECT 
        COUNT(*) as total_analyzed,
        SUM(CASE WHEN prediction = 'hate' THEN 1 ELSE 0 END) as hate_count,
        SUM(CASE WHEN prediction = 'non-hate' THEN 1 ELSE 0 END) as safe_count,
        ROUND(AVG(confidence) * 100, 2) as avg_confidence,
        COUNT(DISTINCT DATE(timestamp)) as active_days
    FROM memes;
END //

-- Procedure to clean old records
CREATE PROCEDURE IF NOT EXISTS CleanOldRecords(IN days_old INT)
BEGIN
    DELETE FROM memes 
    WHERE timestamp < DATE_SUB(NOW(), INTERVAL days_old DAY);
END //

DELIMITER ;


-- ============================================
-- Views
-- ============================================

-- Daily statistics view
CREATE OR REPLACE VIEW daily_stats AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total,
    SUM(CASE WHEN prediction = 'hate' THEN 1 ELSE 0 END) as hate_count,
    SUM(CASE WHEN prediction = 'non-hate' THEN 1 ELSE 0 END) as safe_count,
    ROUND(AVG(confidence) * 100, 2) as avg_confidence
FROM memes
GROUP BY DATE(timestamp)
ORDER BY date DESC;


-- Recent predictions view
CREATE OR REPLACE VIEW recent_predictions AS
SELECT 
    id,
    filename,
    original_filename,
    LEFT(extracted_text, 100) as text_preview,
    prediction,
    ROUND(confidence * 100, 2) as confidence_pct,
    timestamp
FROM memes
ORDER BY timestamp DESC
LIMIT 50;
