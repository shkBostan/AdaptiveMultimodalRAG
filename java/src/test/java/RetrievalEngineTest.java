import com.adaptiveRAG.retrieval.RetrievalEngine;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test cases for RetrievalEngine.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */
public class RetrievalEngineTest {
    
    @Test
    public void testInitialization() {
        RetrievalEngine engine = new RetrievalEngine(768);
        assertNotNull(engine);
    }
}

