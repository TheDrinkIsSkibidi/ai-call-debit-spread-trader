import openai
import anthropic
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from datetime import datetime

from src.core.config import settings
from src.spread_constructor.spread_builder import CallDebitSpread
from src.data_ingestion.market_data import MarketSnapshot


@dataclass
class LLMThesisScore:
    confidence_score: float  # 0-100
    reasoning: str
    key_factors: List[str]
    risk_assessment: str
    market_context: str
    recommendation: str  # BUY, PASS, STRONG_BUY


class LLMThesisScorer:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
        self.anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        # Threshold for filtering trades
        self.min_confidence_threshold = 70
    
    def score_spread_setup(self, spread: CallDebitSpread, market_data: MarketSnapshot, 
                          model: str = "gpt-4") -> LLMThesisScore:
        """Score a call debit spread setup using LLM reasoning"""
        
        # Generate comprehensive prompt
        prompt = self._generate_scoring_prompt(spread, market_data)
        
        if model.startswith("gpt"):
            response = self._score_with_openai(prompt, model)
        elif model.startswith("claude"):
            response = self._score_with_anthropic(prompt, model)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        return self._parse_llm_response(response)
    
    def batch_score_spreads(self, spreads: List[CallDebitSpread], 
                           market_snapshots: Dict[str, MarketSnapshot],
                           model: str = "gpt-4") -> List[Tuple[CallDebitSpread, LLMThesisScore]]:
        """Score multiple spreads and return only those above threshold"""
        scored_spreads = []
        
        for spread in spreads:
            market_data = market_snapshots.get(spread.symbol)
            if not market_data:
                continue
            
            try:
                score = self.score_spread_setup(spread, market_data, model)
                
                # Only include spreads above confidence threshold
                if score.confidence_score >= self.min_confidence_threshold:
                    scored_spreads.append((spread, score))
                    
            except Exception as e:
                print(f"Error scoring spread for {spread.symbol}: {e}")
                continue
        
        # Sort by confidence score (highest first)
        scored_spreads.sort(key=lambda x: x[1].confidence_score, reverse=True)
        return scored_spreads
    
    def _generate_scoring_prompt(self, spread: CallDebitSpread, market_data: MarketSnapshot) -> str:
        """Generate comprehensive prompt for LLM analysis"""
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
You are an expert options trader specializing in call debit spreads. Analyze the following trade setup and provide a confidence score (0-100) with detailed reasoning.

**Current Date**: {current_date}

**Stock Information**:
- Symbol: {spread.symbol}
- Current Price: ${market_data.price:.2f}
- RSI: {market_data.rsi:.1f if market_data.rsi else 'N/A'}
- 8-period EMA: ${market_data.ema_8:.2f if market_data.ema_8 else 'N/A'}
- 21-period EMA: ${market_data.ema_21:.2f if market_data.ema_21 else 'N/A'}
- IV Rank: {market_data.iv_rank:.1f if market_data.iv_rank else 'N/A'}%
- Volume: {market_data.volume:,}

**Call Debit Spread Setup**:
- Expiration: {spread.expiration}
- Days to Expiration: {spread.days_to_expiration}
- Long Strike: ${spread.long_strike:.2f} (Delta: {spread.long_option.delta:.3f})
- Short Strike: ${spread.short_strike:.2f} (Delta: {spread.short_option.delta:.3f})
- Net Debit: ${spread.net_debit:.2f}
- Max Profit: ${spread.max_profit:.2f}
- Max Loss: ${spread.max_loss:.2f}
- Breakeven: ${spread.breakeven:.2f}
- Risk/Reward Ratio: {spread.risk_reward_ratio:.2f}
- Profit Probability: {spread.profit_probability:.1%}

**Key Analysis Points**:
1. **Trend Analysis**: Is the stock in a favorable trend for this bullish trade?
2. **Technical Setup**: How do the technical indicators support or contradict this trade?
3. **Option Structure**: Are the strikes, deltas, and expiration optimal?
4. **Risk Management**: Is the risk/reward attractive and appropriate?
5. **Market Environment**: How does current IV rank and market conditions affect this trade?
6. **Timing**: Is this an optimal entry point?

**Required Response Format**:
Please respond in the following JSON format:

{{
    "confidence_score": [0-100 integer],
    "reasoning": "[2-3 paragraph detailed analysis]",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_assessment": "[assessment of primary risks]",
    "market_context": "[current market environment analysis]",
    "recommendation": "[BUY/PASS/STRONG_BUY]"
}}

**Scoring Guidelines**:
- 90-100: Exceptional setup with multiple confirming factors
- 80-89: Strong setup with favorable risk/reward
- 70-79: Good setup worth considering
- 60-69: Marginal setup with mixed signals
- Below 60: Poor setup, should be avoided

Consider factors like trend strength, technical confirmation, option structure efficiency, market timing, and overall probability of success.
"""
        return prompt
    
    def _score_with_openai(self, prompt: str, model: str = "gpt-4") -> str:
        """Get scoring response from OpenAI"""
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert options trader with 20+ years of experience trading call debit spreads. Provide objective, data-driven analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise
    
    def _score_with_anthropic(self, prompt: str, model: str = "claude-3-sonnet-20240229") -> str:
        """Get scoring response from Anthropic Claude"""
        try:
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=1500,
                temperature=0.3,
                system="You are an expert options trader with 20+ years of experience trading call debit spreads. Provide objective, data-driven analysis.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            raise
    
    def _parse_llm_response(self, response: str) -> LLMThesisScore:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                return LLMThesisScore(
                    confidence_score=float(data.get('confidence_score', 0)),
                    reasoning=data.get('reasoning', ''),
                    key_factors=data.get('key_factors', []),
                    risk_assessment=data.get('risk_assessment', ''),
                    market_context=data.get('market_context', ''),
                    recommendation=data.get('recommendation', 'PASS')
                )
            else:
                # Fallback parsing if JSON format is not found
                return self._fallback_parse(response)
                
        except json.JSONDecodeError:
            return self._fallback_parse(response)
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return LLMThesisScore(
                confidence_score=0,
                reasoning="Error parsing response",
                key_factors=[],
                risk_assessment="Unknown",
                market_context="Unknown",
                recommendation="PASS"
            )
    
    def _fallback_parse(self, response: str) -> LLMThesisScore:
        """Fallback parsing when JSON format is not detected"""
        # Extract confidence score
        confidence_score = 50  # Default
        
        # Look for patterns like "confidence: 75" or "score: 85"
        import re
        score_patterns = [
            r'confidence[:\s]+(\d+)',
            r'score[:\s]+(\d+)',
            r'(\d+)[/\s]*100',
            r'(\d+)%'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response.lower())
            if match:
                try:
                    confidence_score = int(match.group(1))
                    break
                except:
                    continue
        
        # Extract recommendation
        recommendation = "PASS"
        if any(word in response.lower() for word in ["strong buy", "strong_buy"]):
            recommendation = "STRONG_BUY"
        elif any(word in response.lower() for word in ["buy", "purchase", "take"]):
            recommendation = "BUY"
        elif any(word in response.lower() for word in ["pass", "avoid", "skip"]):
            recommendation = "PASS"
        
        return LLMThesisScore(
            confidence_score=confidence_score,
            reasoning=response[:500],  # First 500 chars as reasoning
            key_factors=[],
            risk_assessment="Parsed from unstructured response",
            market_context="Parsed from unstructured response",
            recommendation=recommendation
        )
    
    def analyze_portfolio_patterns(self, trade_history: List[Dict]) -> Dict:
        """Use LLM to analyze trading patterns and provide insights"""
        
        # Prepare trade history summary
        winning_trades = [t for t in trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trade_history if t.get('pnl', 0) <= 0]
        
        prompt = f"""
Analyze the following trading performance data and identify patterns, strengths, and areas for improvement:

**Performance Summary**:
- Total Trades: {len(trade_history)}
- Winning Trades: {len(winning_trades)}
- Losing Trades: {len(losing_trades)}
- Win Rate: {len(winning_trades)/len(trade_history)*100:.1f}%

**Recent Trades** (last 10):
{json.dumps(trade_history[-10:], indent=2)}

Please provide:
1. Key patterns in winning vs losing trades
2. Behavioral insights and biases detected
3. Specific recommendations for improvement
4. Confidence adjustments for future trades

Format your response as structured analysis with actionable insights.
"""
        
        try:
            response = self._score_with_openai(prompt, "gpt-4")
            return {
                'analysis': response,
                'timestamp': datetime.now().isoformat(),
                'trades_analyzed': len(trade_history)
            }
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return {
                'analysis': 'Error analyzing patterns',
                'timestamp': datetime.now().isoformat(),
                'trades_analyzed': 0
            }