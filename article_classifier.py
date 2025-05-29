import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Clean and preprocess text for comparison"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_keywords_from_interests(interests):
    """Extract individual keywords from research interests"""
    if pd.isna(interests) or interests is None:
        return []
    
    # Split by common delimiters
    keywords = re.split(r'[,;|&\n]+', str(interests))
    
    # Clean each keyword
    cleaned_keywords = []
    stop_words = {'and', 'or', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    
    for keyword in keywords:
        keyword = keyword.strip().lower()
        if keyword and keyword not in stop_words and len(keyword) > 2:
            cleaned_keywords.append(keyword)
    
    return cleaned_keywords

def simple_text_similarity(text1, text2):
    """Simple word overlap similarity"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def calculate_similarity_scores(article_text, research_topic):
    """Calculate similarity between article and research topic"""
    if not article_text or not research_topic:
        return 0.0
    
    # Method 1: Simple word overlap
    word_overlap = simple_text_similarity(article_text, research_topic)
    
    # Method 2: Keyword matching
    topic_words = research_topic.lower().split()
    article_words = article_text.lower().split()
    
    keyword_matches = 0
    for topic_word in topic_words:
        if len(topic_word) > 3:  # Only consider meaningful words
            for article_word in article_words:
                if topic_word in article_word or article_word in topic_word:
                    keyword_matches += 1
                    break
    
    keyword_score = keyword_matches / len(topic_words) if topic_words else 0.0
    
    # Method 3: Exact phrase matching
    phrase_score = 1.0 if research_topic.lower() in article_text.lower() else 0.0
    
    # Combine scores
    final_score = (word_overlap * 0.4) + (keyword_score * 0.4) + (phrase_score * 0.2)
    return min(final_score, 1.0)

def classify_articles_to_topics():
    """Main function to classify articles to topics for each researcher"""
    
    print("🚀 Starting article classification...")
    
    # Load researcher data
    try:
        researchers_df = pd.read_csv('all_researchers_summary.csv')
        print(f"✅ Loaded {len(researchers_df)} researchers")
        print(f"📋 Columns: {list(researchers_df.columns)}")
    except Exception as e:
        print(f"❌ Error loading researchers: {e}")
        return
    
    classification_results = []
    researcher_corpora_path = 'researcher_corpora'
    
    if not os.path.exists(researcher_corpora_path):
        print(f"❌ Error: {researcher_corpora_path} folder not found")
        return
    
    # Create backup directory
    backup_path = 'researcher_corpora_backup'
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
        print(f"📁 Created backup directory: {backup_path}")
    
    processed_count = 0
    
    for index, researcher_row in researchers_df.iterrows():
        researcher_name = researcher_row['Researcher Name']
        research_interests = researcher_row.get('Research Interests', '')
        
        if pd.isna(researcher_name) or not researcher_name:
            continue
            
        if pd.isna(research_interests) or not research_interests:
            print(f"⚠️  No research interests for {researcher_name}")
            continue
        
        print(f"\n📖 Processing: {researcher_name}")
        print(f"🔬 Interests: {research_interests}")
        
        # Find researcher's articles file
        # Try different filename formats
        possible_files = [
            f"{researcher_name}_publications.csv",
            f"{researcher_name.replace(' ', '_')}_publications.csv",
            f"{researcher_name.replace(' ', '_').replace('.', '')}_publications.csv"
        ]
        
        researcher_file = None
        for filename in possible_files:
            file_path = os.path.join(researcher_corpora_path, filename)
            if os.path.exists(file_path):
                researcher_file = file_path
                break
        
        if not researcher_file:
            print(f"   ❌ No articles file found")
            continue
        try:
            articles_df = pd.read_csv(researcher_file)
            print(f"   📚 Found {len(articles_df)} articles")
            
            # Create backup of original file
            backup_file = os.path.join(backup_path, os.path.basename(researcher_file))
            articles_df.to_csv(backup_file, index=False)
            
        except Exception as e:
            print(f"   ❌ Error reading articles: {e}")
            continue
        
        # Extract topics from research interests
        topics = extract_keywords_from_interests(research_interests)
        if not topics:
            print(f"   ⚠️  No valid topics extracted")
            continue
        
        print(f"   🏷️  Topics: {topics}")
        
        # Add classification columns if they don't exist
        if 'Classified_Topic' not in articles_df.columns:
            articles_df['Classified_Topic'] = 'Unclassified'
        if 'Classification_Confidence' not in articles_df.columns:
            articles_df['Classification_Confidence'] = 0.0
        
        article_count = 0
        
        # Process each article
        for idx, article_row in articles_df.iterrows():
            # Get article data
            title = str(article_row.get('Title', ''))
            description = str(article_row.get('Description', ''))
            journal = str(article_row.get('Journal', ''))
            date = str(article_row.get('Publication date', ''))
            
            # Combine text for classification
            article_text = f"{title} {description}"
            article_text = preprocess_text(article_text)
            
            if len(article_text) < 10:  # Skip very short articles
                continue
            
            # Find best matching topic
            best_topic = None
            best_score = 0.0
            threshold = 0.05  # Lower threshold for better matching
            
            for topic in topics:
                score = calculate_similarity_scores(article_text, topic)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_topic = topic
            
            # Update the dataframe with classification results
            classified_topic = best_topic if best_topic else 'Unclassified'
            articles_df.at[idx, 'Classified_Topic'] = classified_topic
            articles_df.at[idx, 'Classification_Confidence'] = round(best_score, 3)
            
            # Store result for summary
            result = {
                'Researcher_Name': researcher_name,
                'Article_Title': title,
                'Article_Description': description[:200] + '...' if len(description) > 200 else description,
                'Publication_Date': date,
                'Journal': journal,
                'Classified_Topic': classified_topic,
                'Confidence_Score': round(best_score, 3),
                'All_Research_Interests': research_interests
            }
            
            classification_results.append(result)
            article_count += 1
          # Save updated articles back to the corpus file
        try:
            articles_df.to_csv(researcher_file, index=False, encoding='utf-8')
            print(f"   💾 Updated corpus file with classifications")
        except PermissionError:
            # File might be open, try with backup name
            backup_name = researcher_file.replace('.csv', '_classified.csv')
            try:
                articles_df.to_csv(backup_name, index=False, encoding='utf-8')
                print(f"   💾 Saved classifications to: {backup_name}")
                print(f"   ⚠️  Original file was locked, created new file")
            except Exception as e:
                print(f"   ❌ Could not save classifications: {e}")
        except Exception as e:
            print(f"   ⚠️  Error saving updated corpus: {e}")
        
        print(f"   ✅ Processed {article_count} articles")
        processed_count += 1
        
        # Show progress every 10 researchers
        if processed_count % 10 == 0:
            print(f"\n📊 Progress: {processed_count} researchers processed...")
      # Save results
    if classification_results:
        results_df = pd.DataFrame(classification_results)
        
        # Try to save with timestamp if file is locked
        base_filename = 'articles_topic_classification'
        output_file = f'{base_filename}.csv'
        
        try:
            results_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\n🎉 Classification Complete!")
            print(f"📊 Total articles processed: {len(results_df)}")
            print(f"📁 Results saved to: {output_file}")
        except PermissionError:
            # File is open elsewhere, use timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f'{base_filename}_{timestamp}.csv'
            try:
                results_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f"\n🎉 Classification Complete!")
                print(f"📊 Total articles processed: {len(results_df)}")
                print(f"📁 Results saved to: {output_file}")
                print(f"⚠️  Original file was locked, created new file with timestamp")
            except Exception as e:
                print(f"\n❌ Error saving results: {e}")
                print("📊 Classification completed but couldn't save summary file")
                return
        
        # Calculate statistics
        classified_count = len(results_df[results_df['Classified_Topic'] != 'Unclassified'])
        unclassified_count = len(results_df[results_df['Classified_Topic'] == 'Unclassified'])
        
        print(f"\n📈 Results Summary:")
        print(f"   ✅ Successfully classified: {classified_count}")
        print(f"   ❌ Unclassified: {unclassified_count}")
        print(f"   📊 Success rate: {(classified_count/len(results_df)*100):.1f}%")
        
        if classified_count > 0:
            avg_confidence = results_df[results_df['Classified_Topic'] != 'Unclassified']['Confidence_Score'].mean()
            print(f"   🎯 Average confidence: {avg_confidence:.3f}")
            
            # Show top topics
            top_topics = results_df[results_df['Classified_Topic'] != 'Unclassified']['Classified_Topic'].value_counts().head(10)
            print(f"\n🔥 Top 10 Topics:")
            for i, (topic, count) in enumerate(top_topics.items(), 1):
                print(f"   {i:2d}. {topic}: {count} articles")
        
        # Show some examples
        print(f"\n📋 Sample Classifications:")
        sample_classified = results_df[results_df['Classified_Topic'] != 'Unclassified'].head(3)
        for _, row in sample_classified.iterrows():
            print(f"   📄 {row['Article_Title'][:50]}...")
            print(f"      👤 {row['Researcher_Name']}")
            print(f"      🏷️  {row['Classified_Topic']} (confidence: {row['Confidence_Score']})")
            print()
        
    else:
        print("❌ No articles were classified")

def analyze_classification_results():
    """Analyze the classification results"""
    try:
        df = pd.read_csv('articles_topic_classification.csv')
    except:
        print("❌ No classification results found. Run classify_articles_to_topics() first!")
        return
    
    print("📊 Classification Analysis")
    print("=" * 50)
    
    # Per researcher analysis
    researcher_stats = df.groupby('Researcher_Name').agg({
        'Classified_Topic': lambda x: (x != 'Unclassified').sum(),
        'Confidence_Score': 'mean'
    }).round(3)
    
    researcher_stats.columns = ['Classified_Articles', 'Avg_Confidence']
    researcher_stats = researcher_stats.sort_values('Classified_Articles', ascending=False)
    
    print(f"\n🔬 Top 10 Researchers by Classified Articles:")
    print(researcher_stats.head(10))
    
    # Topic distribution
    topic_dist = df[df['Classified_Topic'] != 'Unclassified']['Classified_Topic'].value_counts()
    print(f"\n🏷️  Topic Distribution:")
    print(topic_dist.head(15))

def restore_original_corpus_files():
    """Restore original corpus files from backup"""
    backup_path = 'researcher_corpora_backup'
    corpora_path = 'researcher_corpora'
    
    if not os.path.exists(backup_path):
        print("❌ No backup directory found!")
        return
    
    backup_files = [f for f in os.listdir(backup_path) if f.endswith('.csv')]
    
    if not backup_files:
        print("❌ No backup files found!")
        return
    
    print(f"🔄 Restoring {len(backup_files)} files from backup...")
    
    for file_name in backup_files:
        backup_file = os.path.join(backup_path, file_name)
        original_file = os.path.join(corpora_path, file_name)
        
        try:
            # Read backup and save to original location
            df = pd.read_csv(backup_file)
            df.to_csv(original_file, index=False)
            print(f"   ✅ Restored: {file_name}")
        except Exception as e:
            print(f"   ❌ Error restoring {file_name}: {e}")
    
    print("🎉 Restoration complete!")

def show_classification_sample():
    """Show sample of classifications from corpus files"""
    corpora_path = 'researcher_corpora'
    files = [f for f in os.listdir(corpora_path) if f.endswith('.csv')]
    
    print("\n📋 Sample Classifications from Corpus Files:")
    print("=" * 60)
    
    sample_count = 0
    for file_name in files[:5]:  # Check first 5 files
        file_path = os.path.join(corpora_path, file_name)
        try:
            df = pd.read_csv(file_path)
            if 'Classified_Topic' in df.columns:
                classified = df[df['Classified_Topic'] != 'Unclassified'].head(2)
                if not classified.empty:
                    print(f"\n📁 {file_name.replace('_publications.csv', '')}")
                    for _, row in classified.iterrows():
                        title = str(row.get('Title', ''))[:50]
                        topic = row.get('Classified_Topic', '')
                        confidence = row.get('Classification_Confidence', 0)
                        print(f"   📄 {title}...")
                        print(f"   🏷️  Topic: {topic} (confidence: {confidence})")
                        sample_count += 1
        except Exception as e:
            continue
        
        if sample_count >= 10:  # Limit samples
            break

if __name__ == "__main__":
    print("🔬 Article Classification System")
    print("=" * 50)
    
    # Run classification
    classify_articles_to_topics()
    
    print("\n" + "="*50)
    
    # Analyze results
    analyze_classification_results()
    
    # Show sample classifications from corpus files
    show_classification_sample()
    
    print("\n📁 Files created/updated:")
    print("   📊 articles_topic_classification.csv - Summary of all classifications")
    print("   📚 researcher_corpora/*.csv - Individual files updated with classifications")
    print("   💾 researcher_corpora_backup/*.csv - Backup of original files")
    print("\n💡 To restore original files, call: restore_original_corpus_files()")
    print("\n" + "="*50)
    show_classification_sample()
    print("\n" + "="*50)
    restore_original_corpus_files()
